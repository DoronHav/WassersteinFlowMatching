"""
Scrape particle orientation data from all EMPIAR SPA datasets.

Phase 1: Parse FTP listing → all EMPIAR IDs
Phase 2: Fetch EMPIAR API metadata (parallel) → imageset directories + EMDB links
Phase 3: Filter SPA entries → browse FTP for .star / .cs files (parallel)
Phase 4: Download + parse orientation files → save (N, 3, 3) rotation matrices
Phase 5: Fetch symmetry labels from EMDB (parallel)

Checkpoints after each phase; safe to kill and restart.
Output: /cv/data/braid/havivd/cryodrgn/empiar_scrape/
"""

import os
import re
import json
import time
import pickle
import logging
import hashlib
import requests
import traceback
import numpy as np
from pathlib import Path
from io import BytesIO, StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.transform import Rotation

try:
    import starfile
    HAS_STARFILE = True
except ImportError:
    HAS_STARFILE = False
    print("WARNING: starfile not installed, cannot parse .star files")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
OUT_DIR = Path("/cv/data/braid/havivd/cryodrgn/empiar_scrape")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FTP_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability"
EMPIAR_API = "https://www.ebi.ac.uk/empiar/api/entry"
EMDB_API   = "https://www.ebi.ac.uk/emdb/api/entry"

# Only download files this large or smaller (bytes)
MAX_STAR_SIZE = 600 * 1024 * 1024   # 600 MB

# RELION star file field names
RELION_ROT_FIELDS  = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
RELION_QUAT_FIELD  = "rlnQuaternion"          # rare, some RELION 4 files

N_WORKERS = 16
REQUEST_TIMEOUT = 60

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def session():
    s = requests.Session()
    s.headers.update({"User-Agent": "cryoem-rwfm-scraper/1.0"})
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    s.mount("https://", adapter)
    return s


def safe_get(url, timeout=REQUEST_TIMEOUT, stream=False, **kwargs):
    s = session()
    try:
        r = s.get(url, timeout=timeout, stream=stream, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        log.debug(f"GET failed {url}: {e}")
        return None


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_checkpoint(name, data):
    p = OUT_DIR / f"checkpoint_{name}.json"
    with open(p, "w") as f:
        json.dump(data, f, default=_json_default)
    log.info(f"Checkpoint saved: {p} ({len(data)} items)")


def load_checkpoint(name):
    p = OUT_DIR / f"checkpoint_{name}.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        log.info(f"Checkpoint loaded: {p} ({len(data)} items)")
        return data
    return None


# ──────────────────────────────────────────────
# Phase 1: All EMPIAR IDs from FTP listing
# ──────────────────────────────────────────────

def phase1_get_all_empiar_ids():
    cached = load_checkpoint("phase1_ids")
    if cached:
        return cached

    log.info("Phase 1: fetching EMPIAR FTP directory listing...")
    r = safe_get(f"{FTP_BASE}/")
    if r is None:
        raise RuntimeError("Cannot reach EMPIAR FTP")

    ids = sorted(set(re.findall(r'href="(\d{5,6})/"', r.text)))
    log.info(f"Found {len(ids)} EMPIAR entries in FTP listing")
    save_checkpoint("phase1_ids", ids)
    return ids


# ──────────────────────────────────────────────
# Phase 2: EMPIAR API metadata per entry
# ──────────────────────────────────────────────

def fetch_empiar_meta(empiar_num):
    """Returns dict with keys: empiar_id, imagesets, emdb_ids, experiment_type, title"""
    url = f"{EMPIAR_API}/EMPIAR-{empiar_num}/"
    r = safe_get(url)
    if r is None:
        return None
    try:
        data = r.json()
    except Exception:
        return None

    key = f"EMPIAR-{empiar_num}"
    entry = data.get(key, data.get(key.lower(), None))
    if entry is None:
        # Try first value
        entry = next(iter(data.values()), None)
    if entry is None:
        return None

    imagesets = entry.get("imagesets", [])
    # Extract EMDB cross-references from cross_references field
    cross = entry.get("cross_references", [])
    emdb_ids = [c for c in cross if isinstance(c, str) and c.startswith("EMD-")]

    return {
        "empiar_id": key,
        "empiar_num": empiar_num,
        "title": entry.get("title", ""),
        "experiment_type": entry.get("experiment_type", ""),
        "imagesets": imagesets,
        "emdb_ids": emdb_ids,
    }


def phase2_fetch_metadata(all_ids):
    cached = load_checkpoint("phase2_meta")
    if cached:
        return {m["empiar_num"]: m for m in cached if m}

    log.info(f"Phase 2: fetching EMPIAR metadata for {len(all_ids)} entries...")
    results = {}

    def worker(num):
        meta = fetch_empiar_meta(num)
        return num, meta

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, num): num for num in all_ids}
        for i, fut in enumerate(as_completed(futures)):
            num, meta = fut.result()
            if meta:
                results[num] = meta
            if (i + 1) % 200 == 0:
                log.info(f"  Phase 2: {i+1}/{len(all_ids)} done, {len(results)} valid")

    save_checkpoint("phase2_meta", list(results.values()))
    return results


# ──────────────────────────────────────────────
# Phase 3: Find star/cs files via FTP browsing
# ──────────────────────────────────────────────

SPA_KEYWORDS = ["single particle", "spa", "singleparticle"]

def looks_like_spa(meta):
    et = (meta.get("experiment_type") or "").lower()
    title = (meta.get("title") or "").lower()
    if any(k in et for k in SPA_KEYWORDS):
        return True
    if any(k in title for k in ["ribosome", "particle", "protein", "complex", "enzyme",
                                  "channel", "receptor", "kinase", "polymerase", "virus"]):
        return True
    return False


def parse_ftp_size(size_str):
    """Convert human-readable FTP size like '27M', '1.2G', '456K' to bytes."""
    size_str = size_str.strip()
    if not size_str or size_str == "-":
        return 0
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if size_str[-1].upper() in units:
        try:
            return int(float(size_str[:-1]) * units[size_str[-1].upper()])
        except ValueError:
            return 0
    try:
        return int(size_str)
    except ValueError:
        return 0


def ftp_list_files(empiar_num, subpath="data", depth=0, max_depth=3):
    """Recursively list FTP directory, return list of (url, size_bytes, ext) for star/cs files."""
    if depth > max_depth:
        return []

    url = f"{FTP_BASE}/{empiar_num}/{subpath}/"
    r = safe_get(url, timeout=30)
    if r is None:
        return []

    found = []
    # Match file rows: <a href="NAME.star">...</a>...SIZE
    # FTP listing format: href in <a>, then date, then size in last <td>
    for match in re.finditer(
        r'<a href="([^"?/][^"]*\.(star|cs))">[^<]*</a>.*?<td[^>]*>\s*([0-9.]+[KMGTkmgt]?)\s*</td>',
        r.text, re.DOTALL
    ):
        fname, ext, size_str = match.group(1), match.group(2).lower(), match.group(3)
        size = parse_ftp_size(size_str)
        file_url = f"{FTP_BASE}/{empiar_num}/{subpath}/{fname}"
        found.append((file_url, size, ext))

    # Recurse into subdirectories (exclude parent/sort links)
    subdirs = re.findall(r'<a href="([A-Za-z0-9][^"?]*/)">', r.text)
    for sd in subdirs:
        found.extend(ftp_list_files(empiar_num, f"{subpath}/{sd.rstrip('/')}", depth + 1, max_depth))

    return found


def phase3_find_star_files(meta_dict):
    cached = load_checkpoint("phase3_files")
    if cached:
        return cached  # list of {empiar_num, url, size, ext}

    # Filter to SPA-looking entries only
    spa_entries = [m for m in meta_dict.values() if looks_like_spa(m)]
    log.info(f"Phase 3: browsing FTP for {len(spa_entries)} SPA entries...")

    all_files = []

    def worker(meta):
        num = meta["empiar_num"]
        files = ftp_list_files(num, depth=0, max_depth=3)
        # Filter to reasonable size star files only (not tiny junk, not huge)
        good = [(url, sz, ext) for url, sz, ext in files
                if 1_000 < sz < MAX_STAR_SIZE]
        return num, meta, good

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, m): m for m in spa_entries}
        for i, fut in enumerate(as_completed(futures)):
            num, meta, files = fut.result()
            for url, sz, ext in files:
                all_files.append({
                    "empiar_num": num,
                    "empiar_id": meta["empiar_id"],
                    "emdb_ids": meta["emdb_ids"],
                    "title": meta["title"],
                    "url": url,
                    "size": sz,
                    "ext": ext,
                })
            if (i + 1) % 100 == 0:
                log.info(f"  Phase 3: {i+1}/{len(spa_entries)} browsed, {len(all_files)} files found")

    log.info(f"Phase 3 done: {len(all_files)} candidate files across "
             f"{len(set(f['empiar_num'] for f in all_files))} entries")
    save_checkpoint("phase3_files", all_files)
    return all_files


# ──────────────────────────────────────────────
# Phase 4: Download and parse orientation files
# ──────────────────────────────────────────────

def euler_to_rotmat(rots_deg, tilts_deg, psis_deg):
    """ZYZ Euler angles (degrees) → (N, 3, 3) rotation matrices."""
    euler = np.stack([rots_deg, tilts_deg, psis_deg], axis=1)
    return Rotation.from_euler("ZYZ", euler, degrees=True).as_matrix().astype(np.float32)


def parse_star(content_bytes):
    """Parse RELION star file bytes → (N, 3, 3) rotation matrices or None."""
    if not HAS_STARFILE:
        return None
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name
        data = starfile.read(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        log.debug(f"starfile.read failed: {e}")
        return None

    # data may be a dict of loops or a single DataFrame
    if isinstance(data, dict):
        # Look for the particles block
        particles = None
        for k, v in data.items():
            if hasattr(v, "columns"):
                cols = set(v.columns)
                if "rlnAngleRot" in cols or "rlnAngleTilt" in cols:
                    particles = v
                    break
        if particles is None:
            return None
    else:
        particles = data

    if not hasattr(particles, "columns"):
        return None

    cols = set(particles.columns)

    if all(f in cols for f in RELION_ROT_FIELDS):
        try:
            rots  = particles["rlnAngleRot"].to_numpy(dtype=float)
            tilts = particles["rlnAngleTilt"].to_numpy(dtype=float)
            psis  = particles["rlnAnglePsi"].to_numpy(dtype=float)
            mats = euler_to_rotmat(rots, tilts, psis)
            log.debug(f"  Parsed {len(mats)} particles from ZYZ Euler star")
            return mats
        except Exception as e:
            log.debug(f"  Euler parse error: {e}")
            return None

    return None


def parse_cs(content_bytes):
    """Parse cryoSPARC .cs file bytes → (N, 3, 3) rotation matrices or None."""
    try:
        cs = np.load(BytesIO(content_bytes))
    except Exception as e:
        log.debug(f"  np.load .cs failed: {e}")
        return None

    # alignments3D/pose is [w, x, y, z]
    if "alignments3D/pose" in cs.dtype.names:
        q = cs["alignments3D/pose"]   # (N, 4) as [w, x, y, z]
        if q.ndim != 2 or q.shape[1] != 4:
            log.debug(f"  alignments3D/pose has unexpected shape {q.shape}, skipping")
            return None
        # scipy expects [x, y, z, w]
        q_xyzw = q[:, [1, 2, 3, 0]].astype(float)
        # Normalize
        norms = np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
        q_xyzw = q_xyzw / np.clip(norms, 1e-8, None)
        mats = Rotation.from_quat(q_xyzw).as_matrix().astype(np.float32)
        log.debug(f"  Parsed {len(mats)} particles from cryoSPARC .cs")
        return mats

    # Some .cs files have "pose" directly
    if "pose" in cs.dtype.names:
        try:
            q = cs["pose"].astype(float)
            if q.ndim == 2 and q.shape[1] == 4:
                norms = np.linalg.norm(q, axis=-1, keepdims=True)
                q = q / np.clip(norms, 1e-8, None)
                mats = Rotation.from_quat(q).as_matrix().astype(np.float32)
                return mats
        except Exception:
            pass

    return None


def download_and_parse(file_info):
    """Download one star/cs file and parse orientations. Returns rotmats or None."""
    url, ext, size = file_info["url"], file_info["ext"], file_info["size"]

    # Stream-download with a size cap
    r = safe_get(url, stream=True, timeout=120)
    if r is None:
        return None

    chunks = []
    downloaded = 0
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        downloaded += len(chunk)
        if downloaded > MAX_STAR_SIZE:
            log.debug(f"  File too large, aborting: {url}")
            return None

    content = b"".join(chunks)

    if ext == "star":
        return parse_star(content)
    elif ext == "cs":
        return parse_cs(content)
    return None


def pick_best_file(files_for_entry):
    """
    From multiple candidate files for one EMPIAR entry, pick the best one.
    Heuristic: prefer star files with 'particles' in name; prefer larger files
    (more particles) but not gigantic ones.
    """
    def score(f):
        name = f["url"].lower()
        s = 0
        if "particle" in name: s += 100
        if "run_data" in name: s += 80
        if "shiny" in name: s += 60       # CTF-refined star = final product
        if "consensus" in name: s += 50
        if "final" in name: s += 40
        if f["ext"] == "star": s += 20     # prefer RELION over cryoSPARC
        # Prefer bigger files (more particles), but penalise >200MB
        mb = f["size"] / 1e6
        s += min(mb, 200) / 10             # up to +20
        return s

    return sorted(files_for_entry, key=score, reverse=True)[0]


def phase4_download_parse(file_list):
    """Download + parse one file per EMPIAR entry. Saves .pkl per entry."""
    done_dir = OUT_DIR / "rotations"
    done_dir.mkdir(exist_ok=True)

    # Group files by empiar_num and pick best per entry
    by_entry = {}
    for f in file_list:
        num = f["empiar_num"]
        by_entry.setdefault(num, []).append(f)

    log.info(f"Phase 4: {len(by_entry)} EMPIAR entries with candidate files")

    results = []  # {empiar_id, n_particles, pkl_path, ...}

    def worker(num, candidates):
        out_path = done_dir / f"{num}.pkl"
        if out_path.exists():
            try:
                with open(out_path, "rb") as fh:
                    saved = pickle.load(fh)
                saved_meta = {k: v for k, v in saved.items() if k != "rotations"}
                return {"empiar_num": num, "status": "cached", **saved_meta}
            except Exception:
                pass

        best = pick_best_file(candidates)
        log.info(f"  [{num}] downloading {best['url'].split('/')[-1]} ({best['size']//1024//1024} MB)")
        mats = download_and_parse(best)

        if mats is None or len(mats) < 100:
            return {"empiar_num": num, "status": "no_orientations", "url": best["url"]}

        meta = {
            "empiar_id": best["empiar_id"],
            "empiar_num": num,
            "emdb_ids": best["emdb_ids"],
            "title": best["title"],
            "source_url": best["url"],
            "n_particles": len(mats),
        }
        with open(out_path, "wb") as fh:
            pickle.dump({"rotations": mats, **meta}, fh)

        log.info(f"  [{num}] saved {len(mats)} particles → {out_path.name}")
        return {"empiar_num": num, "status": "ok", "pkl": str(out_path), **meta}

    with ThreadPoolExecutor(max_workers=4) as ex:  # fewer workers for downloads
        futures = {ex.submit(worker, num, cands): num for num, cands in by_entry.items()}
        for i, fut in enumerate(as_completed(futures)):
            try:
                r = fut.result()
            except Exception as e:
                num = futures[fut]
                log.warning(f"  [{num}] worker exception: {e}")
                r = {"empiar_num": num, "status": "error", "error": str(e)}
            results.append(r)
            if (i + 1) % 10 == 0:
                ok = sum(1 for x in results if x.get("status") == "ok")
                log.info(f"Phase 4: {i+1}/{len(by_entry)} done, {ok} successful")

    save_checkpoint("phase4_results", results)
    return results


# ──────────────────────────────────────────────
# Phase 5: Symmetry labels from EMDB
# ──────────────────────────────────────────────

def fetch_symmetry(emdb_id):
    """Fetch symmetry point group for one EMDB entry."""
    r = safe_get(f"{EMDB_API}/{emdb_id}")   # no trailing slash
    if r is None:
        return None
    try:
        data = r.json()
    except Exception:
        return None

    # Navigate nested structure
    try:
        for sd in data.get("structure_determination_list", {}).get("structure_determination", []):
            for ip in sd.get("image_processing", []):
                sym = (ip.get("final_reconstruction", {})
                          .get("applied_symmetry", {})
                          .get("point_group"))
                if sym:
                    return sym
    except Exception:
        pass
    return None


def phase5_symmetry(phase4_results):
    cached = load_checkpoint("phase5_symmetry")
    if cached:
        return {r["empiar_num"]: r for r in cached}

    ok_entries = [r for r in phase4_results if r.get("status") in ("ok", "cached")]
    log.info(f"Phase 5: fetching symmetry for {len(ok_entries)} entries...")

    symmetry_map = {}

    def worker(entry):
        num = entry["empiar_num"]
        emdb_ids = entry.get("emdb_ids", [])
        sym = None
        for eid in emdb_ids:
            sym = fetch_symmetry(eid)
            if sym:
                break
        return num, sym

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, e): e for e in ok_entries}
        for i, fut in enumerate(as_completed(futures)):
            num, sym = fut.result()
            symmetry_map[num] = sym
            if (i + 1) % 50 == 0:
                found = sum(1 for v in symmetry_map.values() if v)
                log.info(f"  Phase 5: {i+1}/{len(ok_entries)}, {found} with symmetry")

    # Attach symmetry back to results
    enriched = []
    for entry in ok_entries:
        num = entry["empiar_num"]
        enriched.append({**entry, "symmetry": symmetry_map.get(num)})

    save_checkpoint("phase5_symmetry", enriched)
    return {r["empiar_num"]: r for r in enriched}


# ──────────────────────────────────────────────
# Final: build manifest
# ──────────────────────────────────────────────

def build_manifest(symmetry_results):
    manifest = []
    for num, entry in symmetry_results.items():
        pkl = OUT_DIR / "rotations" / f"{num}.pkl"
        if pkl.exists():
            manifest.append({
                "empiar_id": entry.get("empiar_id", f"EMPIAR-{num}"),
                "empiar_num": num,
                "emdb_ids": entry.get("emdb_ids", []),
                "title": entry.get("title", ""),
                "symmetry": entry.get("symmetry"),
                "n_particles": entry.get("n_particles"),
                "pkl_path": str(pkl),
            })

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    sym_counts = {}
    for m in manifest:
        sym = m["symmetry"] or "unknown"
        sym_counts[sym] = sym_counts.get(sym, 0) + 1

    log.info(f"\n{'='*50}")
    log.info(f"DONE: {len(manifest)} proteins with orientation data")
    log.info(f"Symmetry distribution: {dict(sorted(sym_counts.items()))}")
    log.info(f"Manifest saved to: {manifest_path}")
    return manifest


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting EMPIAR orientation scraper")
    log.info(f"Output directory: {OUT_DIR}")

    ids        = phase1_get_all_empiar_ids()
    meta_dict  = phase2_fetch_metadata(ids)
    file_list  = phase3_find_star_files(meta_dict)
    p4_results = phase4_download_parse(file_list)
    sym_map    = phase5_symmetry(p4_results)
    manifest   = build_manifest(sym_map)
