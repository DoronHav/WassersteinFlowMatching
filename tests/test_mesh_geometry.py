"""Offline tests for the triangular-mesh general geometry (utils_Mesh.TriangleMesh).

Uses a procedurally generated icosphere (no external assets / network) so the mesh spectral
geometry can be sanity-checked against the analytic sphere.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from wassersteinflowmatching.riemannian_wasserstein import utils_Geom, utils_Mesh


@pytest.fixture(scope="module")
def ico():
    V, F = utils_Mesh.icosphere(n_subdiv=3)  # unit sphere, ~1280 faces
    return V, F


@pytest.fixture(scope="module")
def mesh(ico):
    V, F = ico
    # normalize=False: keep it a unit sphere so we can compare to the analytic sphere
    return utils_Mesh.TriangleMesh(mesh=(V, F), k=40, normalize=False,
                                   n_interpolation_steps=20)


@pytest.fixture(scope="module")
def sphere_ref():
    return utils_Geom.sphere()


def rand_sphere(key, n):
    p = jax.random.normal(key, (n, 3))
    return p / jnp.linalg.norm(p, axis=-1, keepdims=True)


# ---------- projection ----------

class TestProjection:
    def test_project_stays_on_unit_sphere(self, mesh):
        pts = np.array(rand_sphere(jax.random.key(0), 20)) * 1.3  # push off-surface
        proj = np.array(mesh.project_to_geometry(jnp.asarray(pts)))
        norms = np.linalg.norm(proj, axis=-1)
        # icosphere triangles are slightly inside the unit sphere; radius should be ~1
        np.testing.assert_allclose(norms, 1.0, atol=0.02)

    def test_project_cpu_matches_jax(self, mesh):
        pts = np.array(rand_sphere(jax.random.key(1), 10)) * 0.7
        proj_cpu = mesh.project_to_geometry(pts, use_cpu=True)
        proj_jax = np.array(mesh.project_to_geometry(jnp.asarray(pts)))
        np.testing.assert_allclose(proj_cpu, proj_jax, atol=1e-4)

    def test_project_handles_batched_shape(self, mesh):
        pts = np.array(rand_sphere(jax.random.key(2), 12)).reshape(3, 4, 3) * 1.1
        proj = mesh.project_to_geometry(jnp.asarray(pts))
        assert proj.shape == (3, 4, 3)


# ---------- spectral distance ----------

class TestSpectralDistance:
    def test_self_distance_zero(self, mesh):
        p = mesh.project_to_geometry(rand_sphere(jax.random.key(3), 1)[0])
        assert float(mesh._squared_distance(p, p)) < 1e-8

    def test_symmetric_and_positive(self, mesh):
        p = mesh.project_to_geometry(rand_sphere(jax.random.key(4), 1)[0])
        q = mesh.project_to_geometry(rand_sphere(jax.random.key(5), 1)[0])
        d_pq = float(mesh._squared_distance(p, q))
        d_qp = float(mesh._squared_distance(q, p))
        assert d_pq > 0
        np.testing.assert_allclose(d_pq, d_qp, rtol=1e-4)

    def test_correlates_with_geodesic(self, mesh, sphere_ref):
        """Biharmonic spectral distance should be monotonically related to geodesic distance."""
        pts = mesh.project_to_geometry(rand_sphere(jax.random.key(6), 40))
        anchor = pts[0]
        spec = np.array([float(mesh._squared_distance(anchor, q)) for q in pts[1:]])
        geo = np.array([float(sphere_ref.distance(anchor, q)) for q in pts[1:]])  # squared geodesic
        # Spearman rank correlation
        rs = np.corrcoef(np.argsort(np.argsort(spec)), np.argsort(np.argsort(geo)))[0, 1]
        assert rs > 0.9, f"rank correlation {rs} too low"


# ---------- inherited log/exp/interpolant ----------

class TestDifferentialOps:
    def test_velocity_is_tangent(self, mesh):
        p = mesh.project_to_geometry(rand_sphere(jax.random.key(7), 1)[0])
        q = mesh.project_to_geometry(rand_sphere(jax.random.key(8), 1)[0])
        v = mesh.velocity(p, q, 0.0)
        # tangent to the sphere means roughly orthogonal to the radial direction
        radial = p / jnp.linalg.norm(p)
        cos = float(jnp.dot(v / (jnp.linalg.norm(v) + 1e-9), radial))
        assert abs(cos) < 0.1, f"velocity not tangent, cos={cos}"

    def test_interpolant_endpoints(self, mesh):
        p = mesh.project_to_geometry(rand_sphere(jax.random.key(9), 1)[0])
        q = mesh.project_to_geometry(rand_sphere(jax.random.key(10), 1)[0])
        i0 = mesh.interpolant(p, q, 0.0)
        i1 = mesh.interpolant(p, q, 1.0)
        np.testing.assert_allclose(np.array(i0), np.array(p), atol=1e-3)
        np.testing.assert_allclose(np.array(i1), np.array(q), atol=0.1)

    def test_interpolant_and_exp_on_surface(self, mesh):
        p = mesh.project_to_geometry(rand_sphere(jax.random.key(11), 1)[0])
        q = mesh.project_to_geometry(rand_sphere(jax.random.key(12), 1)[0])
        for t in [0.25, 0.5, 0.75]:
            pt = mesh.interpolant(p, q, t)
            assert abs(float(jnp.linalg.norm(pt)) - 1.0) < 0.05
        v = mesh.velocity(p, q, 0.0)
        pe = mesh.exponential_map(p, v, 0.1)
        assert abs(float(jnp.linalg.norm(pe)) - 1.0) < 0.05


# ---------- base distribution ----------

class TestUniformSampling:
    def test_uniform_on_surface(self, mesh):
        pts = mesh.sample_uniform((4, 50, 3), jax.random.key(13))
        assert pts.shape == (4, 50, 3)
        norms = np.array(jnp.linalg.norm(pts, axis=-1))
        np.testing.assert_allclose(norms, 1.0, atol=0.05)


# ---------- end-to-end ----------

class TestFlowMatchingIntegration:
    def test_mesh_wfm_train_and_generate(self, ico):
        from wassersteinflowmatching.riemannian_wasserstein import MeshWassersteinFlowMatching

        V, F = ico
        np.random.seed(0)
        sizes = np.random.randint(low=8, high=16, size=16)
        # random ambient point clouds; the model projects them to the mesh on init
        point_clouds = [np.random.normal(size=[n, 3]) for n in sizes]

        model = MeshWassersteinFlowMatching(
            point_clouds=point_clouds,
            mesh=(V, F),
            k=30,
            normalize=False,
            n_interpolation_steps=10,
            monge_map="sample",
            mini_batch_ot_mode=False,
            num_sinkhorn_iters=50,
            noise_type="uniform_mesh",
        )
        model.train(training_steps=3, batch_size=4, decay_steps=2)
        samples, weights = model.generate_samples(num_samples=4, timesteps=10)

        final = np.array(samples[-1])
        w = np.array(weights)
        norms = np.linalg.norm(final, axis=-1)
        np.testing.assert_allclose(norms[w > 0], 1.0, atol=0.05)
