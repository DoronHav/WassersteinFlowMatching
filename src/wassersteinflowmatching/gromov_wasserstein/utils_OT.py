"""
Optimal-transport utilities for Gromov-Wasserstein Flow Matching.

Provides:
  - make_gw_solver      : build an OTT-JAX GromovWasserstein solver
  - gw_cost_cold        : GW cost + coupling, cold start
  - gw_cost_warm        : GW cost + coupling, warm-start from previous coupling
  - gw_grad_cold        : (cost, coupling), gradient w.r.t. X, cold start
  - gw_grad_warm        : (cost, coupling), gradient w.r.t. X, warm start
  - teacher_step_single : Euler integration of the GW gradient flow for one
                          (X0, X1) pair, with warm starts between steps
"""

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore

from ott.geometry import pointcloud  # type: ignore
from ott.geometry import geometry as ott_geometry  # type: ignore
from ott.problems.quadratic import quadratic_problem  # type: ignore
from ott.solvers.linear import sinkhorn  # type: ignore
from ott.solvers.quadratic import gromov_wasserstein  # type: ignore
from ott.initializers.quadratic import initializers as quad_init  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_sq_dist(x: jnp.ndarray) -> jnp.ndarray:
    """Pairwise squared-Euclidean distance matrix for a single point cloud."""
    sq = jnp.sum(x ** 2, axis=-1)
    return sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def make_gw_solver(
    epsilon: float = 0.05,
    lse_mode: bool = True,
) -> gromov_wasserstein.GromovWasserstein:
    """Build an OTT-JAX GromovWasserstein solver.

    The Sinkhorn inner solver uses ``use_danskin=True`` so that
    ``jax.grad`` through the GW cost implements the envelope-theorem
    approximation (gradient with optimal coupling treated as fixed).
    Sinkhorn iterates until its own convergence threshold is met.

    :param epsilon: Entropic regularization strength.
    :param lse_mode: Whether to use log-sum-exp mode in Sinkhorn.
    :return: A configured :class:`GromovWasserstein` solver instance.
    """
    linear_solver = sinkhorn.Sinkhorn(
        lse_mode=lse_mode,
        use_danskin=True,   # envelope-theorem gradient
    )
    return gromov_wasserstein.GromovWasserstein(
        linear_solver=linear_solver,
        epsilon=epsilon,
        warm_start=True,
    )


# ---------------------------------------------------------------------------
# GW cost functions (JAX-differentiable)
# ---------------------------------------------------------------------------

def _build_gw_problem(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: jnp.ndarray = None,
    b: jnp.ndarray = None,
    scale_cost: bool = True,
) -> quadratic_problem.QuadraticProblem:
    """Construct a :class:`QuadraticProblem` from two point clouds.

    When ``scale_cost=True`` the pairwise squared-distance matrices are
    computed explicitly and each is divided by its own maximum value before
    being wrapped in a :class:`~ott.geometry.geometry.Geometry`.  This avoids
    the OTT ``PointCloud`` ``scale_cost`` code path which calls a deprecated
    JAX internal API in the current OTT release.
    """
    if scale_cost:
        Cx = _pairwise_sq_dist(x)
        Cy = _pairwise_sq_dist(y)
        Cx = Cx / (jnp.max(Cx) + 1e-8)
        Cy = Cy / (jnp.max(Cy) + 1e-8)
        geom_x = ott_geometry.Geometry(cost_matrix=Cx)
        geom_y = ott_geometry.Geometry(cost_matrix=Cy)
    else:
        geom_x = pointcloud.PointCloud(x)
        geom_y = pointcloud.PointCloud(y)
    return quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a, b=b)


def gw_cost_cold(
    solver: gromov_wasserstein.GromovWasserstein,
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: jnp.ndarray = None,
    b: jnp.ndarray = None,
    scale_cost: bool = True,
) -> tuple:
    """Regularized GW cost and coupling matrix, cold start.

    :param solver: Pre-built :class:`GromovWasserstein` solver.
    :param x: Source point cloud, shape ``(n, d)``.
    :param y: Target point cloud, shape ``(m, d)``.
    :param a: Source marginal weights, shape ``(n,)``; uniform if ``None``.
    :param b: Target marginal weights, shape ``(m,)``; uniform if ``None``.
    :param scale_cost: Divide intra-geometry cost matrices by their max.
    :return: ``(reg_gw_cost, coupling)`` where coupling has shape ``(n, m)``.
    """
    prob = _build_gw_problem(x, y, a=a, b=b, scale_cost=scale_cost)
    out = solver(prob)
    return out.reg_gw_cost, out.matrix


def gw_cost_warm(
    solver: gromov_wasserstein.GromovWasserstein,
    epsilon: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    prev_coupling: jnp.ndarray,
    a: jnp.ndarray = None,
    b: jnp.ndarray = None,
    scale_cost: bool = True,
) -> tuple:
    """Regularized GW cost and coupling matrix, warm-started from *prev_coupling*.

    :param solver: Pre-built :class:`GromovWasserstein` solver.
    :param epsilon: Regularization strength (must match solver).
    :param x: Source point cloud, shape ``(n, d)``.
    :param y: Target point cloud, shape ``(m, d)``.
    :param prev_coupling: Coupling from the previous GW solve, shape ``(n, m)``.
    :param a: Source marginal weights, shape ``(n,)``; uniform if ``None``.
    :param b: Target marginal weights, shape ``(m,)``; uniform if ``None``.
    :param scale_cost: Divide intra-geometry cost matrices by their max.
    :return: ``(reg_gw_cost, coupling)`` where coupling has shape ``(n, m)``.
    """
    prob = _build_gw_problem(x, y, a=a, b=b, scale_cost=scale_cost)
    initializer = quad_init.QuadraticInitializer(init_coupling=prev_coupling)
    init_lp = initializer(prob, epsilon=epsilon)
    out = solver(prob, init=init_lp)
    return out.reg_gw_cost, out.matrix


# ---------------------------------------------------------------------------
# Envelope-theorem gradients
# ---------------------------------------------------------------------------

def gw_grad_cold(
    solver: gromov_wasserstein.GromovWasserstein,
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: jnp.ndarray = None,
    b: jnp.ndarray = None,
    scale_cost: bool = True,
) -> tuple:
    """GW gradient w.r.t. *x* (cold start) via envelope theorem.

    Uses ``jax.value_and_grad`` with ``has_aux=True`` so the coupling is
    returned without affecting the gradient.

    :param solver: Pre-built :class:`GromovWasserstein` solver.
    :param x: Source point cloud, shape ``(n, d)``.
    :param y: Target point cloud, shape ``(m, d)``.
    :param a: Source marginal weights, shape ``(n,)``; uniform if ``None``.
    :param b: Target marginal weights, shape ``(m,)``; uniform if ``None``.
    :param scale_cost: Divide intra-geometry cost matrices by their max.
    :return: ``((cost, coupling), grad_x)``
    """
    fn = lambda x_: gw_cost_cold(solver, x_, y, a=a, b=b, scale_cost=scale_cost)
    return jax.value_and_grad(fn, has_aux=True)(x)


def gw_grad_warm(
    solver: gromov_wasserstein.GromovWasserstein,
    epsilon: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    prev_coupling: jnp.ndarray,
    a: jnp.ndarray = None,
    b: jnp.ndarray = None,
    scale_cost: bool = True,
) -> tuple:
    """GW gradient w.r.t. *x* (warm start) via envelope theorem.

    :param solver: Pre-built :class:`GromovWasserstein` solver.
    :param epsilon: Regularization strength (must match solver).
    :param x: Source point cloud, shape ``(n, d)``.
    :param y: Target point cloud, shape ``(m, d)``.
    :param prev_coupling: Coupling from the previous GW solve, shape ``(n, m)``.
    :param a: Source marginal weights, shape ``(n,)``; uniform if ``None``.
    :param b: Target marginal weights, shape ``(m,)``; uniform if ``None``.
    :param scale_cost: Divide intra-geometry cost matrices by their max.
    :return: ``((cost, coupling), grad_x)``
    """
    fn = lambda x_: gw_cost_warm(solver, epsilon, x_, y, prev_coupling, a=a, b=b, scale_cost=scale_cost)
    return jax.value_and_grad(fn, has_aux=True)(x)


# ---------------------------------------------------------------------------
# Single-sample teacher step
# ---------------------------------------------------------------------------

def teacher_step_single(
    solver: gromov_wasserstein.GromovWasserstein,
    epsilon: float,
    teacher_dt: float,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: float,
    grad_clip: float = 1.0,
    scale_cost: bool = True,
    w0: jnp.ndarray = None,
    w1: jnp.ndarray = None,
) -> tuple:
    """Compute (X_t, v_t) for a single pair using Euler integration with warm starts.

    Implements the teacher path from the GW-GFM spec:

    1. Compute K = floor(t / teacher_dt) integration steps.
    2. For each step:  solve GW, compute envelope-theorem gradient, Euler update.
    3. Solve GW one final time at X_t to obtain the teacher velocity.

    The GW coupling from step k is passed as a warm start to step k+1.
    The GW gradient is clipped to Frobenius norm ``grad_clip`` at each step to
    prevent divergence when the source and target clouds are far apart.

    :param solver: Pre-built :class:`GromovWasserstein` solver.
    :param epsilon: Regularization strength.
    :param teacher_dt: Euler step size.
    :param x0: Initial (noise) point cloud, shape ``(n, d)``.
    :param x1: Target (data) point cloud, shape ``(m, d)``.
    :param t: Time t ∈ [0, 1]; controls the number of integration steps.
    :param grad_clip: Maximum allowed Frobenius norm of the gradient per step.
    :param scale_cost: Divide intra-geometry cost matrices by their max.
    :return: ``(x_t, v_t)`` where both have shape ``(n, d)``.
    """
    num_steps = int(t / teacher_dt)
    x = x0
    prev_coupling = None

    def _clip(g):
        norm = jnp.linalg.norm(g)
        return jnp.where(norm > grad_clip, g * grad_clip / (norm + 1e-8), g)

    for k in range(num_steps):
        if prev_coupling is None:
            (_, coupling), grad = gw_grad_cold(solver, x, x1, a=w0, b=w1, scale_cost=scale_cost)
        else:
            (_, coupling), grad = gw_grad_warm(solver, epsilon, x, x1, prev_coupling, a=w0, b=w1, scale_cost=scale_cost)
        prev_coupling = jax.lax.stop_gradient(coupling)
        #grad = _clip(grad)
        x = x - teacher_dt * grad

    # Final GW solve at X_t to obtain teacher velocity
    if prev_coupling is None:
        (_, _), grad_final = gw_grad_cold(solver, x, x1, a=w0, b=w1, scale_cost=scale_cost)
    else:
        (_, _), grad_final = gw_grad_warm(solver, epsilon, x, x1, prev_coupling, a=w0, b=w1, scale_cost=scale_cost)

    grad_final = _clip(grad_final)
    x_t = x
    v_t = -grad_final   # velocity = negative gradient of GW energy
    return x_t, v_t
