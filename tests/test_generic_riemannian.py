import pytest
import numpy as np
import jax
import jax.numpy as jnp

from wassersteinflowmatching.riemannian_wasserstein import utils_Geom


@pytest.fixture
def sphere_ref():
    return utils_Geom.sphere()


@pytest.fixture
def mystery():
    return utils_Geom.MysterySphere(n_interpolation_steps=20)


def random_sphere_points(key, n, d=3):
    """Generate n random unit vectors in R^d."""
    pts = jax.random.normal(key, (n, d))
    return pts / jnp.linalg.norm(pts, axis=-1, keepdims=True)


# ---------- distance ----------

class TestDistance:
    def test_distance_matches_sphere(self, sphere_ref, mystery):
        key = jax.random.key(0)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(1), 1, 3)[0]

        d_ref = sphere_ref.distance(p, q)
        d_mys = mystery.distance(p, q)
        np.testing.assert_allclose(float(d_ref), float(d_mys), atol=1e-5)

    def test_distance_matrix_matches_sphere(self, sphere_ref, mystery):
        key = jax.random.key(2)
        P0 = random_sphere_points(key, 5, 3)
        P1 = random_sphere_points(jax.random.key(3), 7, 3)

        dm_ref = sphere_ref.distance_matrix(P0, P1)
        dm_mys = mystery.distance_matrix(P0, P1)
        np.testing.assert_allclose(np.array(dm_ref), np.array(dm_mys), atol=1e-4)


# ---------- log map / velocity ----------

class TestLogMap:
    def test_velocity_direction_matches_sphere(self, sphere_ref, mystery):
        """The autodiff log map should produce a velocity in the same direction
        as the closed-form sphere velocity."""
        key = jax.random.key(10)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(11), 1, 3)[0]

        v_ref = sphere_ref.velocity(p, q, 0.0)
        v_mys = mystery.velocity(p, q, 0.0)

        # Normalize and check cosine similarity
        v_ref_n = v_ref / (jnp.linalg.norm(v_ref) + 1e-9)
        v_mys_n = v_mys / (jnp.linalg.norm(v_mys) + 1e-9)
        cos_sim = float(jnp.dot(v_ref_n, v_mys_n))
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low"

    def test_velocity_magnitude_matches_sphere(self, sphere_ref, mystery):
        key = jax.random.key(12)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(13), 1, 3)[0]

        v_ref = sphere_ref.velocity(p, q, 0.0)
        v_mys = mystery.velocity(p, q, 0.0)

        norm_ref = float(jnp.linalg.norm(v_ref))
        norm_mys = float(jnp.linalg.norm(v_mys))
        np.testing.assert_allclose(norm_ref, norm_mys, rtol=0.05)

    def test_velocity_is_tangent(self, mystery):
        """The autodiff velocity should be tangent to the sphere (orthogonal to p)."""
        key = jax.random.key(14)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(15), 1, 3)[0]

        v = mystery.velocity(p, q, 0.0)
        dot = float(jnp.dot(v, p))
        assert abs(dot) < 1e-4, f"Velocity not tangent: <v,p> = {dot}"


# ---------- exp map ----------

class TestExpMap:
    def test_exp_stays_on_manifold(self, mystery):
        key = jax.random.key(20)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(21), 1, 3)[0]

        v = mystery.velocity(p, q, 0.0)
        p_new = mystery.exponential_map(p, v, 0.1)
        norm = float(jnp.linalg.norm(p_new))
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_exp_small_step_matches_sphere(self, sphere_ref, mystery):
        key = jax.random.key(22)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(23), 1, 3)[0]

        v_ref = sphere_ref.velocity(p, q, 0.0)
        dt = 0.01

        p_ref = sphere_ref.exponential_map(p, v_ref, dt)
        p_mys = mystery.exponential_map(p, v_ref, dt)

        np.testing.assert_allclose(np.array(p_ref), np.array(p_mys), atol=1e-3)


# ---------- interpolant ----------

class TestInterpolant:
    def test_interpolant_endpoints(self, mystery):
        """interpolant(P0, P1, 0) ~ P0 and interpolant(P0, P1, 1) ~ P1."""
        key = jax.random.key(30)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(31), 1, 3)[0]

        interp_0 = mystery.interpolant(p, q, 0.0)
        interp_1 = mystery.interpolant(p, q, 1.0)

        np.testing.assert_allclose(np.array(interp_0), np.array(p), atol=1e-3)
        np.testing.assert_allclose(np.array(interp_1), np.array(q), atol=0.05)

    def test_interpolant_midpoint_matches_sphere(self, sphere_ref, mystery):
        key = jax.random.key(32)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(33), 1, 3)[0]

        mid_ref = sphere_ref.interpolant(p, q, 0.5)
        mid_mys = mystery.interpolant(p, q, 0.5)

        np.testing.assert_allclose(np.array(mid_ref), np.array(mid_mys), atol=0.05)

    def test_interpolant_on_manifold(self, mystery):
        key = jax.random.key(34)
        p = random_sphere_points(key, 1, 3)[0]
        q = random_sphere_points(jax.random.key(35), 1, 3)[0]

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt = mystery.interpolant(p, q, t)
            norm = float(jnp.linalg.norm(pt))
            np.testing.assert_allclose(norm, 1.0, atol=1e-3)


# ---------- weighted mean ----------

class TestWeightedMean:
    def test_weighted_mean_on_manifold(self, mystery):
        key = jax.random.key(40)
        points = random_sphere_points(key, 10, 3)
        weights = jnp.ones(10) / 10
        mean = mystery.weighted_mean(points, weights)
        norm = float(jnp.linalg.norm(mean))
        np.testing.assert_allclose(norm, 1.0, atol=1e-4)

    def test_weighted_mean_matches_sphere(self, sphere_ref, mystery):
        """Karcher mean (iterative log-map) may differ from the simple
        Euclidean-project mean used by the sphere class, but the result
        should still be on the manifold and in a reasonable neighborhood."""
        key = jax.random.key(41)
        # Use tightly clustered points so both methods converge similarly
        base = random_sphere_points(key, 1, 3)[0]
        noise = jax.random.normal(jax.random.key(42), (10, 3)) * 0.1
        points = jax.vmap(mystery.project_to_geometry)(base + noise)
        weights = jnp.ones(10) / 10

        mean_ref = sphere_ref.weighted_mean(points, weights)
        mean_mys = mystery.weighted_mean(points, weights)
        np.testing.assert_allclose(np.array(mean_ref), np.array(mean_mys), atol=0.15)


# ---------- tangent norm ----------

class TestTangentNorm:
    def test_tangent_norm_matches_sphere(self, sphere_ref, mystery):
        key = jax.random.key(50)
        p = random_sphere_points(key, 1, 3)[0]
        q1 = random_sphere_points(jax.random.key(51), 1, 3)[0]
        q2 = random_sphere_points(jax.random.key(52), 1, 3)[0]

        v = sphere_ref.velocity(p, q1, 0.0)
        w = sphere_ref.velocity(p, q2, 0.0)

        tn_ref = sphere_ref.tangent_norm(v, w, p)
        tn_mys = mystery.tangent_norm(v, w, p)
        np.testing.assert_allclose(float(tn_ref), float(tn_mys), atol=1e-4)


# ---------- integration test: full flow matching ----------

class TestFlowMatchingIntegration:
    def test_mystery_sphere_flow_matching(self):
        """End-to-end: create a RWFM model with MysterySphere geometry,
        train for a few steps, and generate samples."""
        from wassersteinflowmatching.riemannian_wasserstein import (
            RiemannianWassersteinFlowMatching,
        )

        np.random.seed(42)
        point_cloud_sizes = np.random.randint(low=8, high=16, size=32)
        point_clouds = [np.random.normal(size=[n, 3]) for n in point_cloud_sizes]

        model = RiemannianWassersteinFlowMatching(
            point_clouds=point_clouds,
            geom="MysterySphere",
            monge_map="sample",
            mini_batch_ot_mode=False,
            num_sinkhorn_iters=50,
        )

        model.train(training_steps=5, batch_size=4, decay_steps=2)
        samples, weights = model.generate_samples(num_samples=4, timesteps=20)

        # Check that generated samples are on the sphere
        final = samples[-1]
        norms = np.array(jnp.linalg.norm(final, axis=-1))
        np.testing.assert_allclose(norms[weights > 0], 1.0, atol=0.05)
