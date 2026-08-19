"""
General-geometry support for Riemannian Wasserstein Flow Matching via triangular meshes.

Implements the general-geometry case of Riemannian Flow Matching (Chen & Lipman, ICLR 2024,
Section 3.3): a spectral premetric on an arbitrary triangular mesh. The heavy differential
geometry (autodiff log map, retraction exponential map, marching interpolant) is inherited from
``utils_Metric.generic_riemannian``. This module only supplies

  * ``project_to_geometry`` : closest point on the mesh surface (differentiable within a triangle),
  * a differentiable squared **spectral distance**
        d(x, y)^2 = sum_i w(lambda_i) (phi_i(x) - phi_i(y))^2
    where phi_i, lambda_i are the smallest Laplace-Beltrami eigenpairs (biharmonic w = lambda^-2 by
    default), computed once with scipy at construction time.

The spectral weights are folded into an embedding ``embed(x)_i = sqrt(w_i) phi_i(x)`` so that the
squared distance is a plain weighted Euclidean distance in the embedding, making the OT
``distance_matrix`` O(k) after a one-shot per-point embed while the autodiff log map still
differentiates through ``embed(p)``.

Preprocessing (mesh loading, cotangent Laplacian, eigendecomposition) is pure numpy/scipy -- no
extra dependencies beyond scipy.
"""

import numpy as np  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import random  # type: ignore

# Re-export so RiemannianWassersteinFlowMatching's
# ``getattr(self._geom_module, 'generic_riemannian', None)`` subclass check finds it.
from wassersteinflowmatching.riemannian_wasserstein.utils_Metric import generic_riemannian  # type: ignore  # noqa: F401


# ##################################################################################################
# Mesh loading
# ##################################################################################################

def load_obj(path):
    """Load a Wavefront .obj mesh. Returns (V (n,3) float32, F (m,3) int32)."""
    verts, faces = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # face entries can be v, v/vt, v/vt/vn, v//vn -- take the vertex index only
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                # obj is 1-indexed; triangulate polygons via a fan
                for j in range(1, len(idx) - 1):
                    faces.append([idx[0] - 1, idx[j] - 1, idx[j + 1] - 1])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_ply(path):
    """Load an ASCII or binary_little_endian .ply mesh. Returns (V (n,3) float32, F (m,3) int32)."""
    with open(path, "rb") as f:
        raw = f.read()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii", errors="replace").splitlines()

    fmt = "ascii"
    n_vert = n_face = 0
    vert_props = []  # list of (name, type)
    section = None
    for line in header:
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "format":
            fmt = toks[1]
        elif toks[0] == "element":
            section = toks[1]
            if section == "vertex":
                n_vert = int(toks[2])
            elif section == "face":
                n_face = int(toks[2])
        elif toks[0] == "property" and section == "vertex":
            vert_props.append((toks[-1], toks[1]))

    ply_np = {
        "float": np.float32, "float32": np.float32, "double": np.float64, "float64": np.float64,
        "uchar": np.uint8, "uint8": np.uint8, "char": np.int8, "int8": np.int8,
        "ushort": np.uint16, "uint16": np.uint16, "short": np.int16, "int16": np.int16,
        "uint": np.uint32, "uint32": np.uint32, "int": np.int32, "int32": np.int32,
    }

    if fmt == "ascii":
        lines = raw[header_end:].decode("ascii").splitlines()
        names = [p[0] for p in vert_props]
        xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
        V = np.empty((n_vert, 3), np.float32)
        for i in range(n_vert):
            vals = lines[i].split()
            V[i] = (float(vals[xi]), float(vals[yi]), float(vals[zi]))
        faces = []
        for i in range(n_vert, n_vert + n_face):
            vals = [int(v) for v in lines[i].split()]
            cnt, idx = vals[0], vals[1:1 + vals[0]]
            for j in range(1, cnt - 1):
                faces.append([idx[0], idx[j], idx[j + 1]])
        return V, np.asarray(faces, np.int32)

    # binary_little_endian
    dt = np.dtype([(name, ply_np[t]) for name, t in vert_props])
    buf = raw[header_end:]
    vdata = np.frombuffer(buf, dtype=dt, count=n_vert)
    V = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float32)
    offset = n_vert * dt.itemsize
    faces = []
    for _ in range(n_face):
        cnt = np.frombuffer(buf, dtype=np.uint8, count=1, offset=offset)[0]
        offset += 1
        idx = np.frombuffer(buf, dtype=np.int32, count=cnt, offset=offset)
        offset += 4 * cnt
        for j in range(1, cnt - 1):
            faces.append([int(idx[0]), int(idx[j]), int(idx[j + 1])])
    return V, np.asarray(faces, np.int32)


def load_mesh(path):
    """Dispatch on file extension."""
    p = path.lower()
    if p.endswith(".obj"):
        return load_obj(path)
    if p.endswith(".ply"):
        return load_ply(path)
    raise ValueError(f"Unsupported mesh format: {path} (expected .obj or .ply)")


def normalize_mesh(V, scale=0.99):
    """Center and rescale vertices so they lie inside the cube [-scale, scale]^3."""
    V = np.asarray(V, np.float32)
    V = V - V.mean(axis=0, keepdims=True)
    m = np.abs(V).max()
    return (V / (m + 1e-12) * scale).astype(np.float32)


def decimate_vertex_cluster(V, F, n_grid=48):
    """Coarsen a mesh by vertex clustering on a regular grid.

    Snaps vertices into an ``n_grid**3`` voxel grid, replaces each occupied cell with the centroid
    of its vertices, remaps faces, and drops degenerate/duplicate faces. Pure numpy; no external
    dependencies. Orientation may be lost (harmless for the cotangent Laplacian and closest-point
    queries). Increase ``n_grid`` for a finer result. Returns (V (n,3) float32, F (m,3) int32).
    """
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    lo, hi = V.min(0), V.max(0)
    cell = np.clip(((V - lo) / (hi - lo + 1e-9) * n_grid).astype(np.int64), 0, n_grid - 1)
    key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
    _, inv = np.unique(key, return_inverse=True)

    n_new = inv.max() + 1
    newV = np.zeros((n_new, 3))
    cnt = np.zeros(n_new)
    np.add.at(newV, inv, V)
    np.add.at(cnt, inv, 1.0)
    newV /= cnt[:, None]

    newF = inv[F]
    good = ((newF[:, 0] != newF[:, 1]) & (newF[:, 1] != newF[:, 2]) & (newF[:, 0] != newF[:, 2]))
    newF = np.unique(np.sort(newF[good], axis=1), axis=0)
    return newV.astype(np.float32), newF.astype(np.int32)


def largest_component(V, F):
    """Keep only the largest connected component of a mesh (drops stray islands).

    Useful after :func:`decimate_vertex_cluster`, which can leave a few disconnected fragments that
    would introduce spurious near-zero Laplacian eigenvalues. Returns (V, F) re-indexed.
    """
    from scipy.sparse import csr_matrix  # type: ignore
    from scipy.sparse.csgraph import connected_components  # type: ignore

    V = np.asarray(V)
    F = np.asarray(F)
    n = V.shape[0]
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    A = csr_matrix((np.ones(len(e0)), (e0, e1)), shape=(n, n))
    A = A + A.T
    _, labels = connected_components(A, directed=False)
    keep = np.bincount(labels).argmax()
    vmask = labels == keep
    remap = -np.ones(n, dtype=np.int64)
    remap[vmask] = np.arange(vmask.sum())
    fmask = vmask[F].all(axis=1)
    newF = remap[F[fmask]]
    return V[vmask].astype(np.float32), newF.astype(np.int32)


def icosphere(n_subdiv=2, radius=1.0):
    """Build a subdivided-icosahedron unit sphere mesh. Returns (V (n,3), F (m,3)).

    Useful as a procedural test manifold (a mesh whose spectral distance can be compared to the
    analytic sphere) and requires no external assets.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    for _ in range(n_subdiv):
        mid_cache = {}
        new_faces = []
        verts = list(verts)

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid_cache:
                mid_cache[key] = len(verts)
                verts.append((np.asarray(verts[a]) + np.asarray(verts[b])) / 2.0)
            return mid_cache[key]

        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(new_faces, dtype=np.int32)

    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return verts.astype(np.float32), faces.astype(np.int32)


# ##################################################################################################
# Laplace-Beltrami spectral basis (numpy / scipy, one-time preprocessing)
# ##################################################################################################

def _cotangent_laplacian(V, F):
    """Cotangent stiffness matrix L and lumped (barycentric) mass matrix M.

    Returns (L csc, M csc, M_diag (n,), face_areas (m,)).
    """
    from scipy import sparse  # type: ignore

    n = V.shape[0]
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]

    def cot(a, b):
        cross = np.cross(a, b)
        return np.sum(a * b, axis=1) / (np.linalg.norm(cross, axis=1) + 1e-12)

    # cotangent of the angle at each vertex; it is the weight of the *opposite* edge
    cot0 = cot(v1 - v0, v2 - v0)  # angle at v0 -> edge (i1, i2)
    cot1 = cot(v2 - v1, v0 - v1)  # angle at v1 -> edge (i2, i0)
    cot2 = cot(v0 - v2, v1 - v2)  # angle at v2 -> edge (i0, i1)

    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    Wv = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    W = sparse.csr_matrix((Wv, (I, J)), shape=(n, n))
    L = sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W

    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    M_diag = np.zeros(n)
    for idx in (i0, i1, i2):
        np.add.at(M_diag, idx, face_areas / 3.0)
    M_diag = np.maximum(M_diag, 1e-12)
    M = sparse.diags(M_diag)
    return L.tocsc(), M.tocsc(), M_diag, face_areas


def _spectral_basis(V, F, k, spectral="biharmonic", tau=1.0):
    """Smallest k non-trivial Laplace-Beltrami eigenpairs and spectral weights.

    Returns (lam (k,), Phi (n,k), w (k,)).
    """
    from scipy.sparse.linalg import eigsh  # type: ignore

    L, M, _, _ = _cotangent_laplacian(V, F)
    L, M = L.astype(np.float64), M.astype(np.float64)  # float64 helps ARPACK convergence
    k = int(min(k, V.shape[0] - 2))
    # shift-invert around 0 to get the smallest eigenvalues of the generalized problem L phi = lam M phi
    vals, vecs = eigsh(L, k=k + 1, M=M, sigma=-1e-8, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    lam = np.maximum(vals[1:k + 1], 1e-12)  # drop the constant (lambda ~ 0) mode
    Phi = vecs[:, 1:k + 1]

    if spectral == "biharmonic":
        w = 1.0 / (lam ** 2)
    elif spectral == "diffusion":
        w = np.exp(-2.0 * tau * lam)
    else:
        raise ValueError(f"Unknown spectral distance '{spectral}' (expected 'biharmonic' or 'diffusion')")
    return lam.astype(np.float32), Phi.astype(np.float32), w.astype(np.float32)


# ##################################################################################################
# Closest point on a triangle (vectorized, works with numpy or jax.numpy)
# ##################################################################################################

def _closest_bary(p, a, b, c, xp):
    """Closest point of ``p`` on triangle (a, b, c), returned as barycentric coords.

    Vectorized over any leading dims that broadcast between p and (a, b, c). ``xp`` is numpy or
    jax.numpy. Returns barycentric weights (..., 3) that sum to 1; the closest point is
    ``bary[...,0]*a + bary[...,1]*b + bary[...,2]*c``. Implements the region test from Ericson,
    "Real-Time Collision Detection".
    """
    eps = 1e-12
    ab, ac, ap = b - a, c - a, p - a
    d1 = xp.sum(ab * ap, axis=-1)
    d2 = xp.sum(ac * ap, axis=-1)
    bp = p - b
    d3 = xp.sum(ab * bp, axis=-1)
    d4 = xp.sum(ac * bp, axis=-1)
    cp = p - c
    d5 = xp.sum(ab * cp, axis=-1)
    d6 = xp.sum(ac * cp, axis=-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = 1.0 / (va + vb + vc + eps)

    # default: interior (face) projection
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w

    def sel(cond, uu, vv, ww, u, v, w):
        return (xp.where(cond, uu, u), xp.where(cond, vv, v), xp.where(cond, ww, w))

    # edge BC
    denbc = (d4 - d3) + (d5 - d6)
    wbc = (d4 - d3) / (denbc + eps)
    u, v, w = sel((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), 0.0, 1.0 - wbc, wbc, u, v, w)
    # edge AC
    wac = d2 / (d2 - d6 + eps)
    u, v, w = sel((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - wac, 0.0, wac, u, v, w)
    # edge AB
    vab = d1 / (d1 - d3 + eps)
    u, v, w = sel((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - vab, vab, 0.0, u, v, w)
    # vertices
    u, v, w = sel((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0, u, v, w)          # C
    u, v, w = sel((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0, u, v, w)          # B
    u, v, w = sel((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0, u, v, w)          # A

    return xp.stack([u, v, w], axis=-1)


# ##################################################################################################
# Triangle mesh geometry
# ##################################################################################################

class TriangleMesh(generic_riemannian):
    """Riemannian geometry on a triangular mesh via a spectral (biharmonic) premetric.

    :param mesh: path to a .obj/.ply file, or a ``(V, F)`` tuple of vertices (n,3) and faces (m,3).
    :param k: number of Laplace-Beltrami eigenfunctions for the spectral distance (default 200).
    :param spectral: 'biharmonic' (w = lambda^-2, tuning-free) or 'diffusion' (w = exp(-2*tau*lambda)).
    :param tau: diffusion-distance time parameter (only used when spectral='diffusion').
    :param normalize: rescale the mesh into [-0.99, 0.99]^3 (default True).
    :param n_interpolation_steps: substeps for the marching geodesic interpolant (default 50).
    :param n_exp_steps: substeps for the retraction exponential map (default: n_interpolation_steps).
    """

    def __init__(self, mesh, k=200, spectral="biharmonic", tau=1.0, normalize=True,
                 n_interpolation_steps=50, n_exp_steps=None):
        super().__init__(n_interpolation_steps=n_interpolation_steps, n_exp_steps=n_exp_steps)

        if isinstance(mesh, str):
            V, F = load_mesh(mesh)
        else:
            V, F = mesh
            V = np.asarray(V, np.float32)
            F = np.asarray(F, np.int32)
        if normalize:
            V = normalize_mesh(V)

        lam, Phi, w = _spectral_basis(V, F, k, spectral=spectral, tau=tau)
        SPhi = Phi * np.sqrt(w)[None, :]  # embed(x)_i = sqrt(w_i) * phi_i(x)

        # numpy copies (CPU projection path)
        self._V = V
        self._F = F
        self._tri = V[F]  # (m, 3, 3)
        self._SPhi = SPhi
        self._Phi = Phi  # raw per-vertex eigenfunctions (n, k)
        self._w = w      # spectral weights (k,)

        # jnp copies (runtime)
        self.V = jnp.asarray(V)
        self.F = jnp.asarray(F)
        self.tri = jnp.asarray(V[F])
        self.tA = self.tri[:, 0]
        self.tB = self.tri[:, 1]
        self.tC = self.tri[:, 2]
        self.SPhi = jnp.asarray(SPhi)
        self.Phi = jnp.asarray(Phi)
        self.w = jnp.asarray(w)
        self.face_areas = jnp.asarray(0.5 * np.linalg.norm(
            np.cross(V[F][:, 1] - V[F][:, 0], V[F][:, 2] - V[F][:, 0]), axis=1))
        self.n_faces = F.shape[0]
        self.lam = jnp.asarray(lam)
        self.k = SPhi.shape[1]

        # Per-triangle unit normals and analytic (FEM P1) gradients of the embedding.
        # The embedding is piecewise-linear per triangle, so grad(sqrt(w_i) phi_i) is a constant
        # tangent vector on each triangle -- well-defined and nonzero even at vertices/edges, where
        # autodiff through the closest-point projection spuriously returns 0.
        e1v, e2v = self._tri[:, 1] - self._tri[:, 0], self._tri[:, 2] - self._tri[:, 0]
        nrm = np.cross(e1v, e2v)
        tri_n = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        Mmat = np.stack([e1v, e2v, tri_n], axis=1)                       # (m, 3, 3) rows
        Minv = np.linalg.inv(Mmat + np.eye(3)[None] * 1e-9)
        E0, E1, E2 = SPhi[F[:, 0]], SPhi[F[:, 1]], SPhi[F[:, 2]]         # (m, k)
        RHS = np.stack([E1 - E0, E2 - E0, np.zeros_like(E0)], axis=1)    # (m, 3, k)
        gradE = np.einsum("mij,mjk->mik", Minv, RHS)                     # (m, 3(spatial), k)
        self.SG = jnp.asarray(np.moveaxis(gradE, 1, 2))                  # (m, k, 3)
        self.tri_n = jnp.asarray(tri_n)

    # ---- location / projection -------------------------------------------------------------

    def _locate(self, p):
        """Nearest triangle to a single point p (3,): returns (idx, barycentric (3,), closest (3,))."""
        bary = _closest_bary(p, self.tA, self.tB, self.tC, jnp)  # (m, 3)
        closest = (bary[:, 0:1] * self.tA + bary[:, 1:2] * self.tB + bary[:, 2:3] * self.tC)
        sqd = jnp.sum((closest - p) ** 2, axis=-1)
        idx = jnp.argmin(sqd)
        return idx, bary[idx], closest[idx]

    def _proj_one(self, p):
        """Closest surface point to a single ambient point p (3,)."""
        return self._locate(p)[2]

    def project_to_geometry(self, P, use_cpu=False):
        if use_cpu:
            return self._project_np(np.asarray(P))
        shape = P.shape
        out = jax.vmap(self._proj_one)(P.reshape(-1, 3))
        return out.reshape(shape)

    def _project_np(self, P, chunk=256):
        shape = P.shape
        Pf = P.reshape(-1, 3)
        tA, tB, tC = self._tri[:, 0], self._tri[:, 1], self._tri[:, 2]
        out = np.empty_like(Pf)
        for s in range(0, Pf.shape[0], chunk):
            p = Pf[s:s + chunk][:, None, :]  # (c, 1, 3)
            bary = _closest_bary(p, tA[None], tB[None], tC[None], np)  # (c, m, 3)
            closest = (bary[..., 0:1] * tA[None] + bary[..., 1:2] * tB[None] + bary[..., 2:3] * tC[None])
            sqd = np.sum((closest - p) ** 2, axis=-1)  # (c, m)
            idx = np.argmin(sqd, axis=1)
            out[s:s + chunk] = closest[np.arange(closest.shape[0]), idx]
        return out.reshape(shape)

    # ---- spectral embedding / distance ------------------------------------------------------

    def _embed(self, p):
        """Differentiable spectral embedding of a single ambient point p (3,) -> (k,)."""
        bary = _closest_bary(p, self.tA, self.tB, self.tC, jnp)  # (m, 3)
        closest = (bary[:, 0:1] * self.tA + bary[:, 1:2] * self.tB + bary[:, 2:3] * self.tC)
        sqd = jnp.sum((closest - p) ** 2, axis=-1)
        idx = jnp.argmin(sqd)
        phi_verts = self.SPhi[self.F[idx]]  # (3, k)
        return bary[idx] @ phi_verts  # (k,)

    def _squared_distance(self, p, q):
        return jnp.sum((self._embed(p) - self._embed(q)) ** 2)

    def eval_vertex_field(self, P, field):
        """Interpolate a per-vertex scalar/vector field at surface points P (..., 3).

        ``field`` has shape (n,) or (n, c). Returns values of shape P.shape[:-1] (+ (c,)).
        Handy for evaluating an eigenfunction (e.g. ``mesh.Phi[:, j]``) at sampled points when
        constructing eigenfunction-based target distributions.
        """
        field = jnp.asarray(field)

        def one(p):
            bary = _closest_bary(p, self.tA, self.tB, self.tC, jnp)
            closest = (bary[:, 0:1] * self.tA + bary[:, 1:2] * self.tB + bary[:, 2:3] * self.tC)
            idx = jnp.argmin(jnp.sum((closest - p) ** 2, axis=-1))
            return bary[idx] @ field[self.F[idx]]

        shape = P.shape[:-1]
        out = jax.vmap(one)(P.reshape(-1, 3))
        return out.reshape(shape + out.shape[1:])

    def distance(self, P0, P1):
        return jnp.nan_to_num(self._squared_distance(P0, P1), nan=0.0)

    # ---- conditional flow (RFM eq. 13 premetric VF) ----------------------------------------
    #
    # The inherited interpolant/velocity assume a geodesic distance where ||grad d|| = 1, so its
    # log-map magnitude equals the geodesic distance. A spectral premetric does not satisfy this,
    # so we integrate the paper's premetric conditional vector field directly (Chen & Lipman 2024,
    # eq. 13, scheduler kappa(t)=1-t). It is magnitude-normalized so the premetric decreases
    # linearly, d(x_t, x1) = (1-t) d(x0, x1), guaranteeing the path reaches x1 at t = 1.

    def _project_to_tangent(self, p, v):
        """Project ambient vector v onto the tangent plane of p's containing triangle.

        Robust replacement for the autodiff-jvp version, which collapses to 0 at vertices/edges.
        """
        n = self.tri_n[self._locate(p)[0]]
        return v - jnp.dot(v, n) * n

    def _grad_half_d2(self, x, x1, E_x1=None):
        """Analytic (FEM) gradient g = grad_x 1/2 d^2(x, x1) and d^2, using x's containing triangle.

        d^2(x, x1) = sum_i (E_i(x) - E_i(x1))^2 with E = sqrt(w) phi piecewise-linear, so
        grad_x 1/2 d^2 = sum_i (E_i(x) - E_i(x1)) grad E_i, and grad E_i is the constant per-triangle
        gradient in ``self.SG`` -- nonzero and well-defined even where x sits on a vertex/edge.
        """
        idx, bary, _ = self._locate(x)
        E_x = bary @ self.SPhi[self.F[idx]]         # (k,)
        if E_x1 is None:
            E_x1 = self._embed(x1)
        diff = E_x - E_x1                            # (k,)
        g = diff @ self.SG[idx]                      # (3,) tangent to triangle idx
        return g, jnp.sum(diff ** 2)

    def _log_map(self, p, q):
        """log_p(q): tangent vector toward q with magnitude d(p, q)."""
        g, d2 = self._grad_half_d2(p, q)
        d = jnp.sqrt(jnp.maximum(d2, 1e-12))
        gn = jnp.sqrt(jnp.sum(g ** 2)) + 1e-12
        return jnp.nan_to_num(-d * g / gn, nan=0.0)

    def _cond_vf(self, x, x1, d0, E_x1=None):
        """Tangent conditional velocity x-dot that decreases d(x, x1) at the constant rate d0."""
        g, d2 = self._grad_half_d2(x, x1, E_x1=E_x1)
        d = jnp.sqrt(jnp.maximum(d2, 1e-12))
        g_sqnorm = jnp.sum(g ** 2) + 1e-12
        # x-dot = -d0 * grad(d) / ||grad(d)||^2 with grad(d) = g / d  =>  -d0 * d * g / ||g||^2
        return jnp.nan_to_num(-d0 * d * g / g_sqnorm, nan=0.0)

    def interpolant(self, P0, P1, t):
        d0 = jnp.sqrt(jnp.maximum(self._squared_distance(P0, P1), 1e-12))
        E1 = self._embed(P1)
        N = self.n_interpolation_steps
        dt = t / N

        def body(x, _):
            v = self._cond_vf(x, P1, d0, E_x1=E1)
            return self.project_to_geometry(x + v * dt), None

        x, _ = jax.lax.scan(body, P0, None, length=N)
        return jnp.nan_to_num(x, nan=0.0)

    def velocity(self, P0, P1, t):
        d0 = jnp.sqrt(jnp.maximum(self._squared_distance(P0, P1), 1e-12))
        x_t = self.interpolant(P0, P1, t)
        return self._cond_vf(x_t, P1, d0)

    def velocity_at_source(self, P0, P1):
        """velocity(P0, P1, 0) without the interpolation scan.

        At t=0 the marching interpolant is a no-op (x_t == P0), so the initial
        conditional velocity is just ``_cond_vf(P0, P1, d0)``. Skips the
        ``n_interpolation_steps`` per-face projection loop -- important for the
        entropic map, which evaluates this for every source/target pair.
        """
        d0 = jnp.sqrt(jnp.maximum(self._squared_distance(P0, P1), 1e-12))
        return self._cond_vf(P0, P1, d0)

    def distance_matrix(self, P0, P1):
        """Efficient O(k) spectral distance matrix: embed once, then weighted sq-Euclidean."""
        E0 = jax.vmap(self._embed)(P0)  # (n0, k)
        E1 = jax.vmap(self._embed)(P1)  # (n1, k)
        sq0 = jnp.sum(E0 ** 2, axis=-1)
        sq1 = jnp.sum(E1 ** 2, axis=-1)
        d2 = sq0[:, None] + sq1[None, :] - 2.0 * (E0 @ E1.T)
        return jnp.nan_to_num(jnp.maximum(d2, 0.0), nan=0.0)

    # ---- tangent chart: lay a 2D pattern (e.g. an MNIST digit) onto the surface ------------

    def tangent_frame_at(self, anchor):
        """Orthonormal tangent frame (e1, e2) and surface anchor at/near ``anchor`` (3,).

        The anchor is projected to the nearest surface point; the frame is built from that
        triangle's normal. Pure numpy (one-time preprocessing). Returns (anchor_np, e1, e2).
        """
        a = np.asarray(self._project_np(np.asarray(anchor, np.float32)[None])[0])
        tA, tB, tC = self._tri[:, 0], self._tri[:, 1], self._tri[:, 2]
        bary = _closest_bary(a[None], tA[None], tB[None], tC[None], np)[0]  # (m, 3)
        closest = bary[:, 0:1] * tA + bary[:, 1:2] * tB + bary[:, 2:3] * tC
        idx = int(np.argmin(np.sum((closest - a) ** 2, axis=1)))
        v0, v1, v2 = self._tri[idx]
        n = np.cross(v1 - v0, v2 - v0)
        n = n / (np.linalg.norm(n) + 1e-12)
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = ref - n * np.dot(ref, n)
        e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(n, e1)
        return a.astype(np.float32), e1.astype(np.float32), e2.astype(np.float32)

    def square_to_surface(self, points2d, anchor, scale=0.5, frame=None, method="exp"):
        """Map 2D points (in roughly [-1,1]^2) onto the surface through a tangent chart at ``anchor``.

        Places the pattern in the tangent plane at ``anchor`` and lays it onto the mesh, either:

        - ``method='exp'``: push along geodesics via the mesh exponential map -- manifold-native and
          fold-free within the injectivity radius (analogous to ``square_to_sphere``). Best on
          smooth regions; can crumple on a coarse/bumpy mesh with a large ``scale``.
        - ``method='project'``: snap each planar point to its nearest surface point
          (``project_to_geometry``) -- a robust flat "decal" that reads cleanly on moderately curved
          regions but can fold where the surface bends away from the tangent plane.

        :param points2d: (N, 2) array of planar coordinates.
        :param anchor: (3,) approximate surface location for the chart center.
        :param scale: size scaling of the chart (geodesic radius for 'exp', ambient for 'project').
        :param frame: optional (e1, e2) tangent basis; computed from the mesh if None.
        :param method: 'exp' (geodesic) or 'project' (closest-point decal).
        :return: (N, 3) surface points.
        """
        points2d = jnp.asarray(points2d)
        a, e1, e2 = self.tangent_frame_at(anchor)
        if frame is not None:
            e1, e2 = frame
        a = jnp.asarray(a); e1 = jnp.asarray(e1); e2 = jnp.asarray(e2)
        tang = scale * (points2d[:, 0:1] * e1[None, :] + points2d[:, 1:2] * e2[None, :])  # (N, 3)
        if method == "project":
            return self.project_to_geometry(a[None, :] + tang)
        return jax.vmap(lambda v: self.exponential_map(a, v, 1.0))(tang)

    # ---- base distribution: uniform on the surface -----------------------------------------

    def sample_uniform(self, size, key):
        """Sample points uniformly by area over the mesh surface. size = (K, n, 3)."""
        K, n, _ = size
        fkey, bkey = random.split(key)
        probs = self.face_areas / jnp.sum(self.face_areas)
        faces = random.choice(fkey, self.n_faces, shape=(K, n), p=probs)  # (K, n)
        r = random.uniform(bkey, (K, n, 2))
        r1 = jnp.sqrt(r[..., 0])
        b0 = 1.0 - r1
        b1 = r1 * (1.0 - r[..., 1])
        b2 = r1 * r[..., 1]
        tri = self.tri[faces]  # (K, n, 3, 3)
        pts = (b0[..., None] * tri[..., 0, :]
               + b1[..., None] * tri[..., 1, :]
               + b2[..., None] * tri[..., 2, :])
        return pts
