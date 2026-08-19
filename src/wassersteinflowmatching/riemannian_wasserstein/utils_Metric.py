import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np


class generic_riemannian:
    """
    Generic Riemannian geometry where only the squared distance function
    and projection are provided. Log/exp maps are derived automatically
    via autodiff of the distance and ODE-style retraction.

    Subclass this and override:
      - _squared_distance(p, q): squared geodesic distance between two single points
      - project_to_geometry(P, use_cpu=False): project points onto the manifold
    """

    def __init__(self, n_interpolation_steps=10, n_exp_steps=None):
        self.n_interpolation_steps = n_interpolation_steps
        self.n_exp_steps = n_exp_steps if n_exp_steps is not None else n_interpolation_steps

    def _squared_distance(self, p, q):
        """Squared geodesic distance between two single points. Override this."""
        raise NotImplementedError

    def project_to_geometry(self, P, use_cpu=False):
        """Project points onto the manifold. Override this."""
        raise NotImplementedError

    def _project_to_tangent(self, p, v):
        """
        Project ambient vector v onto T_p(M) using the Jacobian of the
        projection map. Valid for any embedded manifold where
        project_to_geometry is differentiable.
        """
        _, v_tangent = jax.jvp(
            lambda x: self.project_to_geometry(x), (p,), (v,)
        )
        return jnp.nan_to_num(v_tangent, nan=0.0)

    def _log_map(self, p, q):
        """
        Log map via gradient of squared distance:
          log_p(q) = -proj_{T_p}( grad_p 1/2 d^2(p,q) )
        """
        grad_half_d2 = jax.grad(
            lambda x: 0.5 * self._squared_distance(x, q)
        )(p)
        return -jnp.nan_to_num(self._project_to_tangent(p, grad_half_d2), nan=0.0)

    def distance(self, P0, P1):
        return jnp.nan_to_num(self._squared_distance(P0, P1), nan=0.0)

    def distance_matrix(self, P0, P1):
        return jnp.nan_to_num(jax.vmap(
            jax.vmap(self._squared_distance, in_axes=(None, 0)),
            in_axes=(0, None)
        )(P0, P1), nan=0.0)

    def velocity(self, P0, P1, t):
        """
        Velocity of the geodesic from P0 to P1 at time t, in extrinsic
        coordinates.  Computed as log_{P_t}(P1) / (1 - t) where
        P_t = interpolant(P0, P1, t).
        """
        Pt = self.interpolant(P0, P1, t)
        v = self._log_map(Pt, P1)
        return jnp.nan_to_num(v / jnp.maximum(1.0 - t, 1e-8), nan=0.0)

    def exponential_map(self, p, v, delta_t):
        """
        Multi-step retraction: break the step into N sub-steps,
        re-projecting onto the manifold and transporting the velocity
        to the tangent space at each intermediate point.
        Reduces to the first-order retraction when n_exp_steps=1.
        """
        N = self.n_exp_steps
        sub_dt = delta_t / N

        def body(carry, i):
            current, current_v = carry
            next_point = self.project_to_geometry(current + current_v * sub_dt)
            next_v = self._project_to_tangent(next_point, current_v)
            # Only update when i < N; otherwise keep current state
            active = i < N
            out_point = jnp.where(active, next_point, current)
            out_v = jnp.where(active, next_v, current_v)
            return (out_point, out_v), None

        (result, _), _ = jax.lax.scan(body, (p, v), jnp.arange(self.n_exp_steps))
        return jnp.nan_to_num(result, nan=0.0)

    def interpolant(self, P0, P1, t):
        """
        Geodesic interpolation via multi-step retraction with velocity
        correction at each substep: at each intermediate point, recompute
        the log map toward P1 and take a proportional step.
        """
        N = self.n_interpolation_steps
        step = t / N

        def body(carry, _):
            current, s = carry
            v = self._log_map(current, P1)
            # v points from current to P1 with magnitude = geodesic distance.
            # We want to advance by fraction step/(1-s) of the remaining path.
            remaining = jnp.maximum(1.0 - s, 1e-8)
            scaled_v = v * (step / remaining)
            next_point = self.project_to_geometry(current + scaled_v)
            return (next_point, s + step), None

        (result, _), _ = jax.lax.scan(body, (P0, 0.0), None, length=N)
        
        return jnp.nan_to_num(result,  nan=0.0)

    def tangent_norm(self, v, w, p):
        """
        Squared difference of tangent vectors in the ambient metric.
        Valid when the Riemannian metric is the one induced by the embedding.
        """
        v = self._project_to_tangent(p, v)
        w = self._project_to_tangent(p, w)
        return jnp.nan_to_num(jnp.mean(jnp.square(v - w)), nan=0.0)

    def weighted_mean(self, points, weights):
        """
        Riemannian weighted mean via iterative log-map averaging (Karcher mean).
        """
        weights = weights / (jnp.sum(weights) + 1e-9)
        # Initialize with projected Euclidean weighted mean
        mean = self.project_to_geometry(
            jnp.sum(points * weights[:, None], axis=0)
        )

        def refine(mean, _):
            logs = jax.vmap(self._log_map, in_axes=(None, 0))(mean, points)
            avg_log = jnp.sum(logs * weights[:, None], axis=0)
            return self.project_to_geometry(mean + avg_log), None

        mean, _ = jax.lax.scan(refine, mean, None, length=3)
        return mean


class mystery_sphere(generic_riemannian):
    """
    A sphere geometry where we pretend we only know the squared geodesic
    distance and the projection. All other operations (log map, exp map,
    interpolation, velocity) are derived automatically from autodiff.
    """

    def __init__(self, n_interpolation_steps=100, n_exp_steps=None):
        super().__init__(n_interpolation_steps=n_interpolation_steps, n_exp_steps=n_exp_steps)

    def project_to_geometry(self, P, use_cpu=False):
        if use_cpu:
            return np.nan_to_num(
                P / np.linalg.norm(P, axis=-1, keepdims=True),
                nan=1 / np.sqrt(P.shape[-1])
            )
        return jnp.nan_to_num(
            P / jnp.linalg.norm(P, axis=-1, keepdims=True),
            nan=1 / jnp.sqrt(P.shape[-1])
        )

    def _squared_distance(self, p, q):
        """Squared great-circle distance on the unit sphere."""
        p = self.project_to_geometry(p)
        q = self.project_to_geometry(q)
        dot = jnp.clip(jnp.dot(p, q), -1.0, 1.0)
        return jnp.arccos(dot) ** 2

