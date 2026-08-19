"""Bunny geometry + MNIST-on-bunny cloud construction (mirrors tutorial_wfm_bunny_mnist.ipynb)."""

import os
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

from wassersteinflowmatching.riemannian_wasserstein import utils_Mesh

BUNNY_URL = ('https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/'
             'master/data/stanford-bunny.obj')

# Fixed data-construction seeds so the train / test clouds are identical across all jobs
# (only the model training key varies with the benchmark seed).
TRAIN_DATA_SEED = 0
TEST_DATA_SEED = 12345


def build_geometry(bunny_path, n_grid=32, k=100, spectral='biharmonic',
                   normalize=False, n_interpolation_steps=100):
    """Load, decimate, and normalise the bunny; return (geom, V, F)."""
    os.makedirs(os.path.dirname(bunny_path) or '.', exist_ok=True)
    if not os.path.exists(bunny_path):
        urllib.request.urlretrieve(BUNNY_URL, bunny_path)
    V, F = utils_Mesh.load_obj(bunny_path)
    V, F = utils_Mesh.decimate_vertex_cluster(V, F, n_grid=n_grid)
    V, F = utils_Mesh.largest_component(V, F)
    V = utils_Mesh.normalize_mesh(V)
    geom = utils_Mesh.TriangleMesh(mesh=(V, F), k=k, spectral=spectral,
                                   normalize=normalize, n_interpolation_steps=n_interpolation_steps)
    return geom, np.array(geom.V), np.array(geom.F)


def chart_frame(geom, V, F):
    """Deterministic tangent-frame chart on the bunny flank (same as the notebook)."""
    Vn = V
    fn = np.cross(Vn[F[:, 1]] - Vn[F[:, 0]], Vn[F[:, 2]] - Vn[F[:, 0]])
    vn = np.zeros_like(Vn)
    for j in range(3):
        np.add.at(vn, F[:, j], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
    ylo, yhi = np.percentile(Vn[:, 1], [20, 55])
    body = np.where((Vn[:, 1] > ylo) & (Vn[:, 1] < yhi))[0]
    anchor = Vn[body[np.argmax(Vn[body, 2])]]            # flank facing +z
    a, e1, e2 = geom.tangent_frame_at(anchor)
    return jnp.asarray(a), jnp.asarray(e1), jnp.asarray(e2)


def _cloud_builder(geom, frame, scale=1.0):
    a_j, e1_j, e2_j = frame

    @jax.jit
    def cloud_to_bunny(pc2d):                             # (K, 2) -> (K, 3)
        tang = scale * (pc2d[:, 0:1] * e1_j[None] + pc2d[:, 1:2] * e2_j[None])
        return jax.vmap(lambda v: geom.exponential_map(a_j, v, 1.0))(tang)

    return jax.jit(jax.vmap(cloud_to_bunny))              # (B, K, 2) -> (B, K, 3)


def _image_to_pc2d(img, n, rng):
    import skimage
    r, c = np.where(img > skimage.filters.threshold_otsu(img))
    if len(r) == 0:
        r, c = np.array([14]), np.array([14])
    pts = np.stack([c, -r], axis=1).astype(np.float32)   # flip row -> upright
    pts = 2 * (pts - pts.min(0)) / (np.ptp(pts, 0) + 1e-9) - 1
    idx = rng.choice(len(pts), size=n, replace=len(pts) < n)
    return pts[idx] + rng.normal(0, 0.02, (n, 2))


def build_digit_clouds(geom, frame, digit, split, n_pts=150, scale=1.0,
                       max_clouds=None, chunk=32):
    """Build MNIST-digit point clouds laid onto the bunny for 'train' or 'test' split.

    :returns: list of (n_pts, 3) numpy arrays.
    """
    import emnist
    if split == 'train':
        imgs, labels = emnist.extract_training_samples('mnist')
        data_seed = TRAIN_DATA_SEED
    elif split == 'test':
        imgs, labels = emnist.extract_test_samples('mnist')
        data_seed = TEST_DATA_SEED
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    imgs = imgs[labels == digit].astype(np.float32) / 255.0
    if max_clouds is not None:
        imgs = imgs[:max_clouds]

    rng = np.random.default_rng(data_seed)
    clouds2d = np.stack([_image_to_pc2d(img, n_pts, rng) for img in imgs])

    batch_to_bunny = _cloud_builder(geom, frame, scale)
    cj = jnp.asarray(clouds2d)
    clouds3d = []
    for s in range(0, cj.shape[0], chunk):
        clouds3d.extend(np.array(batch_to_bunny(cj[s:s + chunk])))
    return clouds3d
