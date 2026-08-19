"""
Expand the EMPIAR orientation scrape beyond the initial 102 proteins.

Two targeted fixes:
  A. Browse the ~1,462 entries skipped by the original SPA filter
     (mostly experiment_type='emdb', which are often SPA datasets cross-listed
     with EMDB structure entries).
  B. Retry the 49 no_orientations failures by trying the top-N candidate
     files instead of just the highest-scored one.

Reads existing checkpoints; appends new results; rebuilds the manifest.
Safe to kill and restart — per-entry pkl caches prevent re-downloading.
"""

import os, re, json, pickle, logging, traceback
import numpy as np
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.transform import Rotation
import requests

try:
    import starfile
    HAS_STARFILE = True
except ImportError:
    HAS_STARFILE = False
    print("WARNING: starfile not installed")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR    = Path("/cv/data/braid/havivd/cryodrgn/empiar_scrape")
DONE_DIR   = OUT_DIR / "rotations"
DONE_DIR.mkdir(exist_ok=True)

FTP_BASE   = "https://ftp.ebi.ac.uk/empiar/world_availability"
EMDB_API   = "https://www.ebi.ac.uk/emdb/api/entry"

MAX_STAR_SIZE = 600 * 1024 * 1024
MIN_PARTICLES = 512
N_WORKERS     = 16
TOP_N_FILES   = 5   # retry top-N candidates for previously failed entries


# ── helpers ──────────────────────────────────────────────────────────────────

def session():
    s = requests.Session()
    s.headers.update({"User-Agent": "cryoem-rwfm-expander/1.0"})
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
    return s

def safe_get(url, timeout=60, stream=False, **kw):
    try:
        r = session().get(url, timeout=timeout, stream=stream, **kw)
        r.raise_for_status()
        return r
    except Exception as e:
        log.debug(f"GET failed {url}: {e}")
        return None

def parse_ftp_size(s):
    s = s.strip()
    if not s or s == "-": return 0
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if s[-1].upper() in units:
        try: return int(float(s[:-1]) * units[s[-1].upper()])
        except ValueError: return 0
    try: return int(s)
    except ValueError: return 0


def euler_to_rotmat(rots, tilts, psis):
    euler = np.stack([rots, tilts, psis], axis=1)
    return Rotation.from_euler("ZYZ", euler, degrees=True).as_matrix().astype(np.float32)


def parse_star(content_bytes):
    if not HAS_STARFILE: return None
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as tmp:
            tmp.write(content_bytes); tmp_path = tmp.name
        data = starfile.read(tmp_path)
        os.unlink(tmp_path)
    except Exception: return None

    df = None
    if isinstance(data, dict):
        for v in data.values():
            if hasattr(v, "columns") and "rlnAngleRot" in v.columns:
                df = v; break
    else:
        df = data if hasattr(data, "columns") else None
    if df is None: return None

    cols = set(df.columns)
    if all(f in cols for f in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]):
        try:
            mats = euler_to_rotmat(
                df["rlnAngleRot"].to_numpy(float),
                df["rlnAngleTilt"].to_numpy(float),
                df["rlnAnglePsi"].to_numpy(float),
            )
            return mats if len(mats) >= MIN_PARTICLES else None
        except Exception: return None
    return None


def parse_cs(content_bytes):
    try: cs = np.load(BytesIO(content_bytes))
    except Exception: return None
    for field in ["alignments3D/pose", "pose"]:
        if field in cs.dtype.names:
            q = cs[field]
            if q.ndim != 2 or q.shape[1] != 4: continue
            if field == "alignments3D/pose":
                q = q[:, [1, 2, 3, 0]]   # wxyz → xyzw
            norms = np.linalg.norm(q.astype(float), axis=-1, keepdims=True)
            q = q.astype(float) / np.clip(norms, 1e-8, None)
            mats = Rotation.from_quat(q).as_matrix().astype(np.float32)
            return mats if len(mats) >= MIN_PARTICLES else None
    return None


def download_and_parse(url, ext):
    r = safe_get(url, stream=True, timeout=120)
    if r is None: return None
    chunks, downloaded = [], 0
    for chunk in r.iter_content(chunk_size=1024*1024):
        chunks.append(chunk); downloaded += len(chunk)
        if downloaded > MAX_STAR_SIZE: return None
    content = b"".join(chunks)
    return parse_star(content) if ext == "star" else parse_cs(content)


def file_score(f):
    name = f["url"].lower(); s = 0
    if "particle" in name: s += 100
    if "run_data"  in name: s += 80
    if "shiny"     in name: s += 60
    if "consensus" in name: s += 50
    if "final"     in name: s += 40
    if f["ext"] == "star":  s += 20
    s += min(f["size"] / 1e6, 200) / 10
    return s


def ftp_list_files(empiar_num, subpath="data", depth=0, max_depth=3):
    if depth > max_depth: return []
    url = f"{FTP_BASE}/{empiar_num}/{subpath}/"
    r = safe_get(url, timeout=30)
    if r is None: return []
    found = []
    for m in re.finditer(
        r'<a href="([^"?/][^"]*\.(star|cs))">[^<]*</a>.*?<td[^>]*>\s*([0-9.]+[KMGTkmgt]?)\s*</td>',
        r.text, re.DOTALL
    ):
        fname, ext, sz = m.group(1), m.group(2).lower(), m.group(3)
        size = parse_ftp_size(sz)
        if 1_000 < size < MAX_STAR_SIZE:
            found.append({"url": f"{FTP_BASE}/{empiar_num}/{subpath}/{fname}",
                          "size": size, "ext": ext})
    for sd in re.findall(r'<a href="([A-Za-z0-9][^"?]*/)">', r.text):
        found.extend(ftp_list_files(empiar_num, f"{subpath}/{sd.rstrip('/')}", depth+1, max_depth))
    return found


def save_entry(num, mats, meta_info):
    out = DONE_DIR / f"{num}.pkl"
    payload = {"rotations": mats, "n_particles": len(mats), **meta_info}
    with open(out, "wb") as f: pickle.dump(payload, f)
    return out


def fetch_symmetry(emdb_id):
    r = safe_get(f"{EMDB_API}/{emdb_id}")
    if r is None: return None
    try: data = r.json()
    except Exception: return None
    try:
        for sd in data.get("structure_determination_list", {}).get("structure_determination", []):
            for ip in sd.get("image_processing", []):
                sym = ip.get("final_reconstruction", {}).get("applied_symmetry", {}).get("point_group")
                if sym: return sym
    except Exception: pass
    return None


# ── Part A: browse un-browsed entries ────────────────────────────────────────

def expand_phase3():
    """Browse entries never seen in the original scrape."""
    p2 = json.load(open(OUT_DIR / "checkpoint_phase2_meta.json"))
    meta_dict = {m["empiar_num"]: m for m in p2 if m}

    p3 = json.load(open(OUT_DIR / "checkpoint_phase3_files.json"))
    already_browsed_nums = set(f["empiar_num"] for f in p3)

    # Also include entries that were browsed but found nothing (need to infer from p4)
    p4 = json.load(open(OUT_DIR / "checkpoint_phase4_results.json"))
    already_processed_nums = set(r["empiar_num"] for r in p4)

    # Entries never browsed: not in phase3 files and not processed
    # We want entries whose pkl doesn't exist (so they haven't been successfully processed)
    done_pkls = {int(p.stem) for p in DONE_DIR.glob("*.pkl")}
    new_entries = [m for num, m in meta_dict.items()
                   if num not in already_browsed_nums and num not in done_pkls]

    log.info(f"Part A: {len(new_entries)} entries to browse for star/cs files")

    new_files = list(p3)  # start from existing

    def worker(meta):
        num = meta["empiar_num"]
        files = ftp_list_files(num)
        return num, meta, files

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, m): m for m in new_entries}
        for i, fut in enumerate(as_completed(futures)):
            num, meta, files = fut.result()
            for f in files:
                new_files.append({
                    "empiar_num": num,
                    "empiar_id": meta.get("empiar_id", f"EMPIAR-{num}"),
                    "emdb_ids": meta.get("emdb_ids", []),
                    "title": meta.get("title", ""),
                    **f,
                })
            if (i + 1) % 200 == 0:
                extra = len(new_files) - len(p3)
                log.info(f"  Part A: {i+1}/{len(new_entries)}, +{extra} new file records")

    # Save updated phase3 checkpoint
    with open(OUT_DIR / "checkpoint_phase3_files.json", "w") as f:
        json.dump(new_files, f)
    log.info(f"Part A done: {len(new_files)} total file records "
             f"(was {len(p3)}, +{len(new_files)-len(p3)} new)")
    return new_files


# ── Part B: retry no_orientations failures ───────────────────────────────────

def retry_failures(all_files):
    """For entries that previously returned no_orientations, try top-N files."""
    p4 = json.load(open(OUT_DIR / "checkpoint_phase4_results.json"))
    no_ori_nums = {r["empiar_num"] for r in p4 if r.get("status") == "no_orientations"}
    # Also skip entries already successfully cached
    done_pkls = {int(p.stem) for p in DONE_DIR.glob("*.pkl")}
    to_retry = no_ori_nums - done_pkls

    log.info(f"Part B: retrying {len(to_retry)} previously failed entries (top {TOP_N_FILES} files each)")

    by_entry = {}
    for f in all_files:
        num = f["empiar_num"]
        if num in to_retry:
            by_entry.setdefault(num, []).append(f)

    recovered = 0

    def worker(num, candidates):
        ranked = sorted(candidates, key=file_score, reverse=True)[:TOP_N_FILES]
        meta_info = {
            "empiar_id": candidates[0].get("empiar_id", f"EMPIAR-{num}"),
            "empiar_num": num,
            "emdb_ids": candidates[0].get("emdb_ids", []),
            "title": candidates[0].get("title", ""),
        }
        for cand in ranked:
            mats = download_and_parse(cand["url"], cand["ext"])
            if mats is not None and len(mats) >= MIN_PARTICLES:
                meta_info["source_url"] = cand["url"]
                save_entry(num, mats, meta_info)
                return num, len(mats), cand["url"]
        return num, 0, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(worker, num, cands): num for num, cands in by_entry.items()}
        for fut in as_completed(futures):
            num, n_particles, url = fut.result()
            if n_particles > 0:
                recovered += 1
                log.info(f"  [{num}] recovered {n_particles} particles from {url.split('/')[-1]}")

    log.info(f"Part B done: recovered {recovered}/{len(to_retry)} previously failed entries")


# ── Part C: process newly found entries ──────────────────────────────────────

def process_new_entries(all_files):
    """Download + parse files for entries not yet in DONE_DIR."""
    done_pkls = {int(p.stem) for p in DONE_DIR.glob("*.pkl")}

    by_entry = {}
    for f in all_files:
        num = f["empiar_num"]
        if num not in done_pkls:
            by_entry.setdefault(num, []).append(f)

    log.info(f"Part C: processing {len(by_entry)} new entries")

    def worker(num, candidates):
        best = sorted(candidates, key=file_score, reverse=True)[0]
        meta_info = {
            "empiar_id": best.get("empiar_id", f"EMPIAR-{num}"),
            "empiar_num": num,
            "emdb_ids": best.get("emdb_ids", []),
            "title": best.get("title", ""),
            "source_url": best["url"],
        }
        mats = download_and_parse(best["url"], best["ext"])
        if mats is None or len(mats) < MIN_PARTICLES:
            return num, 0
        save_entry(num, mats, meta_info)
        return num, len(mats)

    ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(worker, num, cands): num for num, cands in by_entry.items()}
        for i, fut in enumerate(as_completed(futures)):
            num, n = fut.result()
            if n > 0:
                ok += 1
                log.info(f"  [{num}] {n} particles")
            if (i + 1) % 50 == 0:
                log.info(f"  Part C: {i+1}/{len(by_entry)}, {ok} successful so far")

    log.info(f"Part C done: {ok} new entries added")


# ── Part D: fetch symmetry + rebuild manifest ─────────────────────────────────

def rebuild_manifest():
    """Fetch symmetry for any pkl without it; rebuild manifest.json."""
    p5 = json.load(open(OUT_DIR / "checkpoint_phase5_symmetry.json"))
    sym_cache = {r["empiar_num"]: r.get("symmetry") for r in p5}

    all_pkls = list(DONE_DIR.glob("*.pkl"))
    log.info(f"Part D: building manifest from {len(all_pkls)} pkl files")

    # Fetch symmetry for new entries
    need_sym = []
    entries_meta = {}
    for pkl in all_pkls:
        num = int(pkl.stem)
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        entries_meta[num] = d
        if num not in sym_cache:
            need_sym.append((num, d.get("emdb_ids", [])))

    log.info(f"  Fetching symmetry for {len(need_sym)} new entries...")

    def sym_worker(num, emdb_ids):
        for eid in emdb_ids:
            sym = fetch_symmetry(eid)
            if sym: return num, sym
        return num, None

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(sym_worker, num, eids): num for num, eids in need_sym}
        for fut in as_completed(futures):
            num, sym = fut.result()
            sym_cache[num] = sym

    # Build manifest
    manifest = []
    for num, d in entries_meta.items():
        pkl_path = DONE_DIR / f"{num}.pkl"
        manifest.append({
            "empiar_id": d.get("empiar_id", f"EMPIAR-{num}"),
            "empiar_num": num,
            "emdb_ids": d.get("emdb_ids", []),
            "title": d.get("title", ""),
            "symmetry": sym_cache.get(num),
            "n_particles": d.get("n_particles"),
            "pkl_path": str(pkl_path),
        })

    manifest.sort(key=lambda x: x["empiar_num"])
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    from collections import Counter
    sym_dist = Counter(m["symmetry"] or "unknown" for m in manifest)
    log.info(f"Manifest rebuilt: {len(manifest)} proteins")
    log.info(f"Symmetry distribution: {dict(sorted(sym_dist.items()))}")
    return manifest


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== EMPIAR expansion scrape ===")

    log.info("--- Part A: browse un-browsed entries ---")
    all_files = expand_phase3()

    log.info("--- Part B: retry no_orientations failures ---")
    retry_failures(all_files)

    log.info("--- Part C: process newly found entries ---")
    process_new_entries(all_files)

    log.info("--- Part D: rebuild manifest ---")
    manifest = rebuild_manifest()

    log.info(f"Done. Final dataset: {len(manifest)} proteins.")
