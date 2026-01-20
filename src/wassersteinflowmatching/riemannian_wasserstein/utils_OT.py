import jax.numpy as jnp # type: ignore
import ott # type: ignore
from ott.solvers import linear # type: ignore
import jax # type: ignore
from jax import lax # type: ignore
from jax import random # type: ignore
import numpy as np
from tqdm import tqdm


def auto_find_num_iter(point_clouds, weights, eps, lse_mode, distance_matrix_func, num_calc=100, sample_size=2048, noise_point_clouds=None, noise_weights=None, sinkhorn_limit=5000, inner_iterations=10, feature_masks=None, noise_feature_masks=None):
    """
    Find the minimum number of iterations for which at least 80% of OT calculations converge.

    Optimized version using JAX vmap and error tracing.
    Runs a single batched pass to determine the earliest iteration count 
    where 80% of the samples converge.

    Args:
        point_clouds (list): A list of point cloud coordinate arrays.
        weights (list): A list of corresponding weight arrays for each point cloud.
        eps (float): Coefficient of entropic regularization.
        lse_mode (bool): Whether to use log-sum-exp mode.
        distance_matrix_func (callable): Function to compute distance matrix.
        num_calc (int): The number of random pairs to test.
        sample_size (int): The number of points to sample from larger point clouds.
        noise_point_clouds (list, optional): A list of noise point cloud arrays. Defaults to None.
        noise_weights (list, optional): A list of corresponding weights for noise point clouds. Defaults to None.
        sinkhorn_limit (int): Maximum iterations to test.
        inner_iterations (int): Check convergence every N steps.
        feature_masks (list, optional): A list of feature masks for each point cloud. Defaults to None.
        noise_feature_masks (list, optional): A list of feature masks for noise point clouds. Defaults to None.

    Returns:
        int: The recommended number of iterations.
    """
    
    num_clouds = len(point_clouds)
    num_noise_clouds = len(noise_point_clouds) if noise_point_clouds is not None else 0
    
    batch_x_list, batch_y_list = [], []
    batch_a_list, batch_b_list = [], []
    batch_mx_list, batch_my_list = [], []

    max_pc_size = max([len(pc) for pc in point_clouds])
    sample_size = min(sample_size, max_pc_size)

    for _ in range(num_calc):
        # -- Selection Logic --
        if noise_point_clouds is not None and noise_weights is not None:
            idx1 = np.random.randint(0, num_clouds)
            idx2 = np.random.randint(0, num_noise_clouds)
            x, a = point_clouds[idx1], weights[idx1]
            y, b = noise_point_clouds[idx2], noise_weights[idx2]
            mx = feature_masks[idx1] if feature_masks is not None else None
            my = noise_feature_masks[idx2] if noise_feature_masks is not None else None
        else:
            idx1, idx2 = np.random.choice(num_clouds, 2, replace=False)
            x, a = point_clouds[idx1], weights[idx1]
            y, b = point_clouds[idx2], weights[idx2]
            mx = feature_masks[idx1] if feature_masks is not None else None
            my = feature_masks[idx2] if feature_masks is not None else None

        # -- Downsampling / Upsampling Logic --
        # Ensure fixed size for vmap
        
        if len(x) != sample_size:
            replace = len(x) < sample_size
            ix = np.random.choice(len(x), sample_size, replace=replace)
            x, a = x[ix], a[ix]
            if( mx is not None ):
                mx = mx[ix]

        if len(y) != sample_size:
            replace = len(y) < sample_size
            iy = np.random.choice(len(y), sample_size, replace=replace)
            y, b = y[iy], b[iy]
            if( my is not None ):
                my = my[iy]
        # Normalize weights
        a = a / np.sum(a)
        b = b / np.sum(b)
        
        batch_x_list.append(x)
        batch_y_list.append(y)
        batch_a_list.append(a)
        batch_b_list.append(b)
        if mx is not None:
            batch_mx_list.append(mx)
        if my is not None:
            batch_my_list.append(my)

    # Convert to JAX arrays
    batch_x = jnp.array(np.stack(batch_x_list))
    batch_y = jnp.array(np.stack(batch_y_list))
    batch_a = jnp.array(np.stack(batch_a_list))
    batch_b = jnp.array(np.stack(batch_b_list))
    
    if feature_masks is not None:
        batch_mx = jnp.array(np.stack(batch_mx_list))
        batch_my = jnp.array(np.stack(batch_my_list))

    # --------------------------------------------------------------------
    # 2. DEFINING THE VMAP SOLVER (JAX/GPU)
    # --------------------------------------------------------------------
    
    if feature_masks is not None:
        def solve_one(x, y, a, b, mx, my):
            distmat = distance_matrix_func(x, y, mx, my)
            geom = ott.geometry.geometry.Geometry(cost_matrix = distmat, epsilon = eps, scale_cost = 'max_cost')
            
            out = linear.solve(
                geom,
                a=a,
                b=b,
                min_iterations=0, 
                max_iterations=sinkhorn_limit,
                inner_iterations=inner_iterations, 
                lse_mode=lse_mode,
            )
            return out.errors
        
        # JIT compile and VMAP over the 0-th dimension (batch dimension)
        solve_batch = jax.jit(jax.vmap(solve_one, in_axes=(0, 0, 0, 0, 0, 0)))
        # Run the batch computation
        errors = solve_batch(batch_x, batch_y, batch_a, batch_b, batch_mx, batch_my)
    else:
        def solve_one(x, y, a, b):
            distmat = distance_matrix_func(x, y)
            geom = ott.geometry.geometry.Geometry(cost_matrix = distmat, epsilon = eps, scale_cost = 'max_cost')
            
            out = linear.solve(
                geom,
                a=a,
                b=b,
                min_iterations=0, 
                max_iterations=sinkhorn_limit,
                inner_iterations=inner_iterations, 
                lse_mode=lse_mode,
            )
            return out.errors

        # JIT compile and VMAP over the 0-th dimension (batch dimension)
        solve_batch = jax.jit(jax.vmap(solve_one, in_axes=(0, 0, 0, 0)))
    
        # Run the batch computation
        errors = solve_batch(batch_x, batch_y, batch_a, batch_b)

    # --------------------------------------------------------------------
    # 3. ANALYZING THE TRACE
    # --------------------------------------------------------------------

    # OTT pads errors with -1. Real errors are positive.
    # Convergence is when error < threshold (default 1e-3 in linear.solve) OR when it's already finished (-1).
    threshold = 1e-3 
    
    # If error is -1, it means it finished in a previous step (converged).
    # If error is > 0 and < threshold, it converged at this step.
    is_converged = (errors == -1) | ((errors > -1) & (errors < threshold))
    
    # Calculate convergence rate per column (time step)
    convergence_rates = jnp.mean(is_converged.astype(float), axis=0)
    
    # Find the first index where rate >= 0.8
    valid_indices = convergence_rates >= 0.95
    
    if not jnp.any(valid_indices):
        print(f"WARNING: Convergence rate never reached 95% (Max rate: {jnp.max(convergence_rates):.2f}).")
        return sinkhorn_limit

    first_success_index = jnp.argmax(valid_indices)
    
    # Convert index back to iteration count
    recommended_iter = (first_success_index + 1) * inner_iterations
    
    print(f"INFO: Found sufficient convergence (95%) at {recommended_iter} iterations.")
    
    return int(recommended_iter)


def auto_find_num_iter_matrix(distance_matrix, eps=0.01, lse_mode=True, sinkhorn_limit=5000, inner_iterations=10):
    # distance_matrix shape: (Batch, N, N) or (N, N)
    
    # Ensure batch dimension
    if distance_matrix.ndim == 2:
        distance_matrix = distance_matrix[None, :, :]
        
    def solve_one(dist_mat):
            geom = ott.geometry.geometry.Geometry(cost_matrix=dist_mat, epsilon=eps, scale_cost='max_cost')
            out = linear.solve(
            geom,
            min_iterations=0, 
            max_iterations=sinkhorn_limit,
            inner_iterations=inner_iterations, 
            lse_mode=lse_mode,
        )
            return out.errors

    solve_batch = jax.jit(jax.vmap(solve_one))
    errors = solve_batch(distance_matrix) # (Batch, Steps)

    threshold = 1e-3
    is_converged = (errors == -1) | ((errors > -1) & (errors < threshold))
    
    # Calculate convergence rate per column (time step)
    convergence_rates = jnp.mean(is_converged.astype(float), axis=0)
    
    # Find the first index where rate >= 0.8
    valid_indices = convergence_rates >= 0.95
    
    if not jnp.any(valid_indices):
            print(f"WARNING: Convergence rate never reached 95% (Max rate: {jnp.max(convergence_rates):.2f}).")
            return sinkhorn_limit

    first_success_index = jnp.argmax(valid_indices)
    recommended_iter = (first_success_index + 1) * inner_iterations
    return int(recommended_iter)


def auto_find_num_iter_minibatch(point_clouds, weights, noise_point_clouds, noise_weights, ot_mat_jit, eps, lse_mode, num_batches=16, batch_size_ot=32, sample_size=512, key=None, feature_masks=None, noise_feature_masks=None):
    if key is None:
        key = random.key(0)
    
    subkey1, subkey2, key = random.split(key, 3)
    
    total_pc = num_batches * batch_size_ot
    
    # Sample real point clouds
    idx = random.choice(subkey1, point_clouds.shape[0], shape=(total_pc,))
    real_pcs = point_clouds[idx] 
    real_weights = weights[idx]
    
    real_masks = None
    if feature_masks is not None:
        real_masks = feature_masks[idx]

    # Sample noise point clouds
    idx_noise = random.choice(subkey2, noise_point_clouds.shape[0], shape=(total_pc,))
    noise_pcs = noise_point_clouds[idx_noise]
    noise_weights_batch = noise_weights[idx_noise]
    
    noise_masks = None
    if noise_feature_masks is not None:
        noise_masks = noise_feature_masks[idx_noise]
    
    # Downsampling logic
    current_n_points = point_clouds.shape[1]
    sample_size = min(sample_size, current_n_points)

    if sample_size is not None and sample_size != current_n_points:
        replace = current_n_points < sample_size
        
        def get_samples(pc, w, k):
            # pc: (N, D), w: (N,)
            idx = random.choice(k, pc.shape[0], shape=(sample_size,), replace=replace)
            # Normalize weights
            new_w = w[idx]
            new_w = new_w / jnp.sum(new_w)

            return pc[idx], new_w

        key, subkey = random.split(key)
        keys = random.split(subkey, total_pc)
        real_pcs, real_weights = jax.vmap(get_samples)(real_pcs, real_weights, keys)


        key, subkey = random.split(key)
        keys = random.split(subkey, total_pc)
        noise_pcs, noise_weights_batch = jax.vmap(get_samples)(noise_pcs, noise_weights_batch, keys)

    real_pcs_reshaped = real_pcs.reshape(num_batches, batch_size_ot, *real_pcs.shape[1:])
    real_weights_reshaped = real_weights.reshape(num_batches, batch_size_ot, *real_weights.shape[1:])
    noise_pcs_reshaped = noise_pcs.reshape(num_batches, batch_size_ot, *noise_pcs.shape[1:])
    noise_weights_reshaped = noise_weights_batch.reshape(num_batches, batch_size_ot, *noise_weights_batch.shape[1:])
    
    real_masks_reshaped = None
    if real_masks is not None:
        real_masks_reshaped = real_masks.reshape(num_batches, batch_size_ot, *real_masks.shape[1:])
        real_masks_reshaped = real_masks_reshaped[:, :, :real_pcs_reshaped.shape[2], :]
    
    noise_masks_reshaped = None
    if noise_masks is not None:
        noise_masks_reshaped = noise_masks.reshape(num_batches, batch_size_ot, *noise_masks.shape[1:])
        noise_masks_reshaped = noise_masks_reshaped[:, :, :real_pcs_reshaped.shape[2], :]

    if real_masks is not None and noise_masks is not None:

        print(f"pc shapes for auto sinkhorn matrix: {real_pcs_reshaped.shape}, noise shapes: {noise_pcs_reshaped.shape}")
        print(f"weight shapes for auto sinkhorn matrix: {real_weights_reshaped.shape}, noise weight shapes: {noise_weights_reshaped.shape}")
        print(f"mask shapes for auto sinkhorn matrix: {real_masks_reshaped.shape}, noise mask shapes: {noise_masks_reshaped.shape}")

        def compute_batch_op(operand):
            pcs, p_w, p_m, ncs, n_w, n_m = operand
            
            def compute_row_op(i):
                p_i = pcs[i]
                w_i = p_w[i]
                m_i = p_m[i]
                
                p_batch = jnp.broadcast_to(p_i[None, ...], ncs.shape)
                w_batch = jnp.broadcast_to(w_i[None, ...], n_w.shape)
                m_batch = jnp.broadcast_to(m_i[None, ...], n_m.shape)
                
                return ot_mat_jit([p_batch, w_batch, m_batch], [ncs, n_w, n_m])
                
            return jax.lax.map(compute_row_op, jnp.arange(batch_size_ot))

        D = jax.lax.map(compute_batch_op, (real_pcs_reshaped, real_weights_reshaped, real_masks_reshaped, noise_pcs_reshaped, noise_weights_reshaped, noise_masks_reshaped))
    else:

        def compute_batch_op(operand):
            pcs, p_w, ncs, n_w = operand
            
            def compute_row_op(i):
                p_i = pcs[i]
                w_i = p_w[i]
                
                p_batch = jnp.broadcast_to(p_i[None, ...], ncs.shape)
                w_batch = jnp.broadcast_to(w_i[None, ...], n_w.shape)
                
                return ot_mat_jit([p_batch, w_batch], [ncs, n_w])
                
            return jax.lax.map(compute_row_op, jnp.arange(batch_size_ot))

        D = jax.lax.map(compute_batch_op, (real_pcs_reshaped, real_weights_reshaped, noise_pcs_reshaped, noise_weights_reshaped))

    return auto_find_num_iter_matrix(D, eps=eps, lse_mode=lse_mode)


def get_assignments_sampling(P, key, pc_y=None):
    """
    Get assignments by sampling from the OT matrix rows.
    (JAX-compatible version)

    For each real source point, this treats the corresponding row in P as a
    probability distribution and samples a single target point. Returns coordinates or indices.

    Args:
        P: The (padded) optimal transport matrix.
        key: A JAX random key for sampling.
        pc_y: The target point cloud coordinates. If None, returns only indices.

    Returns:
        If pc_y is provided: (coordinates, assignments) where coordinates are target points
        If pc_y is None: (None, assignments) where assignments are indices
    """
    row_sums = jnp.sum(P, axis=1)
    P_normalized = P / (row_sums[:, None] + 1e-9)

    # Generate a key for each source point (row)
    keys = jax.random.split(key, P.shape[0])

    def _sample_row(probs, is_real, subkey):
        """Samples a single target index for one source point."""
        # If the row is not real (a padded point), return -1.
        # Otherwise, sample from the target distribution.
        return jax.lax.cond(
            is_real,
            lambda: jax.random.choice(subkey, a=probs.shape[0], p=probs),
            lambda: -1
        )

    # Vmap the sampling function across all rows.
    is_real_rows = row_sums > 1e-7
    assignments = jax.vmap(_sample_row)(P_normalized, is_real_rows, keys)
    
    # Return coordinates if pc_y is provided, otherwise just assignments
    if pc_y is None:
        return None, assignments
    else:
        return pc_y[assignments.astype(jnp.int32)], assignments

def get_assignments_rounding(P, pc_y=None):
    """
    Get assignments using a greedy rounding scheme (one-to-one).
    (JAX-compatible version)

    This function iteratively finds the highest value in the transport matrix,
    assigns the corresponding points, and removes them from consideration.
    Returns assigned coordinates or indices.

    Args:
        P: The (padded) optimal transport matrix.
        pc_y: The target point cloud coordinates. If None, returns only indices.
        
    Returns:
        If pc_y is provided: (coordinates, assignments) where coordinates are target points
        If pc_y is None: (None, assignments) where assignments are indices
    """
    # Infer which rows/cols correspond to real vs. padded points.
    is_real_row = P.sum(axis=1) > 1e-7
    
    # The number of assignments to make is fixed at compile time for `fori_loop`.
    # We loop for the maximum possible number of assignments.
    num_iter = min(P.shape[0], P.shape[1])

    def loop_body(i, state):
        used_rows, used_cols, assignments = state

        # Mask P to only consider valid, unassigned entries.
        valid_entries_mask = ~used_rows[:, None] & ~used_cols[None, :]
        P_masked = jnp.where(valid_entries_mask, P, -1.0)
        
        # Find the best available assignment.
        row_idx, col_idx = jnp.unravel_index(jnp.argmax(P_masked), P.shape)
        
        # This conditional update is the JAX-friendly way to handle state
        # changes inside a loop.
        return jax.lax.cond(
            P_masked[row_idx, col_idx] >= 0,
            # If a valid assignment was found, update the state.
            lambda: (
                used_rows.at[row_idx].set(True),
                used_cols.at[col_idx].set(True),
                assignments.at[row_idx].set(col_idx)
            ),
            # Otherwise, return the state unchanged.
            lambda: state
        )

    # Initialize the state for the loop.
    # `used_rows/cols` are True if the point has been assigned or is padded.
    init_state = (
        ~is_real_row, # Initially, only padded rows are "used".
        ~(P.sum(axis=0) > 1e-7), # Initially, only padded cols are "used".
        jnp.full(P.shape[0], -1, dtype=jnp.int32) # Assignments start as -1.
    )

    final_state = jax.lax.fori_loop(0, num_iter, loop_body, init_state)

    assignments = final_state[2]
    
    # Return coordinates if pc_y is provided, otherwise just assignments
    if pc_y is None:
        return None, assignments
    else:
        return pc_y[assignments.astype(jnp.int32)], assignments


def get_assignments_barycentric(P, pc_y, weighted_mean_func):
    """
    Get assignments as the barycentric projection of source points.
    (JAX-compatible version)

    Maps each source point to a weighted average of target points.

    Args:
        P: The (padded) optimal transport matrix.
        pc_y: The (padded) target point cloud coordinates.
        weighted_mean_func: A JAX-compatible function that computes the
                            weighted mean of a set of points.
    
    Returns:
        An array of shape [P.shape[0], num_dims] representing the new
        positions (barycenters) for each source point. Padded points are
        mapped to the zero vector.
    """
    row_sums = jnp.sum(P, axis=1, keepdims=True)
    P_normalized = P / (row_sums + 1e-9)

    # vmap the weighted mean function over each row of the normalized P matrix.
    vmap_barycenter = jax.vmap(
        lambda weights: weighted_mean_func(pc_y, weights)
    )
    
    barycenters = vmap_barycenter(P_normalized)
    return barycenters, 0


def get_assignments_entropic(P, pc_x, pc_y, log_map_func, exp_map_function):
    """
    Get assignments using entropic map estimation (tangent space averaging).
    (JAX-compatible version)

    For each source point x, computes the weighted average of log_x(y) for all y,
    weighted by the transport plan P. Then maps back using exp_x.

    Args:
        P: The (padded) optimal transport matrix. Shape (N, M).
        pc_x: The source point cloud coordinates. Shape (N, D).
        pc_y: The target point cloud coordinates. Shape (M, D).
        log_map_func: Function computing log map (velocity) between two points.
                      Signature: log_map_func(p0, p1, t=0) -> velocity vector.
        exp_map_function: Function computing exponential map.
                          Signature: exp_map_function(p, v, t=1) -> point.

    Returns:
        An array of shape [P.shape[0], num_dims] representing the new
        positions for each source point.
    """
    row_sums = jnp.sum(P, axis=1, keepdims=True)
    P_normalized = P / (row_sums + 1e-9)

    def process_single_point(x, weights):
        # x: (D,)
        # weights: (M,)
        
        # Compute velocities from x to all y in pc_y
        # log_map_func(x, y, 0.0)
        velocities = jax.vmap(lambda y: log_map_func(x, y, 0.0))(pc_y)
        
        # Compute weighted average of velocities
        avg_velocity = jnp.sum(velocities * weights[:, None], axis=0)
        
        # Map back to manifold
        new_point = exp_map_function(x, avg_velocity, 1.0)
        
        return new_point

    assignments = jax.vmap(process_single_point)(pc_x, P_normalized)
    
    return assignments, 0


def transport_plan(pc_x, pc_y, distance_matrix_func, eps = 0.01, lse_mode = False, num_iteration = 200): 
    if len(pc_x) == 3:
        pc_x, w_x, m_x = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]
        m_x = None

    if len(pc_y) == 3:
        pc_y, w_y, m_y = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]
        m_y = None

    if m_x is not None and m_y is not None:
        distmat = distance_matrix_func(pc_x, pc_y, m_x, m_y)
    else:
        distmat = distance_matrix_func(pc_x, pc_y)
    
    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distmat, epsilon = eps, scale_cost = 'max_cost'),
        a = w_x,
        b = w_y,
        min_iterations = num_iteration,
        max_iterations = num_iteration,
        lse_mode = lse_mode)
    
    ot_matrix = ot_solve.matrix
    return(ot_matrix, ot_solve)

def transport_plan_random(pc_x, pc_y): 
    # return a random transport plan, where each point in pc_x is assigned to a random point in pc_y (weighted by w_y)

    if len(pc_x) == 3:
        pc_x, w_x, _ = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]

    if len(pc_y) == 3:
        pc_y, w_y, _ = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]

    ot_matrix = jnp.outer(w_x, w_y)
    return(ot_matrix, 0)


def transport_plan_matched(pc_x, pc_y): 
    # return a random transport plan, where each point in pc_x is assigned to a random point in pc_y (weighted by w_y)

    if len(pc_x) == 3:
        pc_x, w_x, _ = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]

    if len(pc_y) == 3:
        pc_y, w_y, _ = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]

    # transport plan is the identity matrix

    ot_matrix = jnp.eye(pc_x.shape[0])
    return(ot_matrix, 0)


def ot_mat_from_distance(distance_matrix, eps = 0.002, lse_mode = True, num_iteration = 200): 
    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distance_matrix, epsilon = eps, scale_cost = 'max_cost'),
        lse_mode = lse_mode,
        min_iterations = num_iteration,
        max_iterations = num_iteration)
    _, map_ind = get_assignments_rounding(ot_solve.matrix, pc_y=None)
    return(map_ind, ot_solve)


def weighted_mean_and_covariance(pc_x, weights):
    """
    Calculate weighted mean and covariance for a batch of point clouds.
    
    Args:
    pc_x: Array of shape (batch_size, num_points, num_dimensions)
    weights: Array of shape (batch_size, num_points)
    
    Returns:
    weighted_mean: Array of shape (batch_size, num_dimensions)
    weighted_cov: Array of shape (batch_size, num_dimensions, num_dimensions)
    """
    
    # Ensure weights sum to 1 for each point cloud in the batch
    normalized_weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    
    # Calculate weighted mean
    weighted_mean = jnp.sum(pc_x * normalized_weights[:, :, jnp.newaxis], axis=1)
    
    # Calculate weighted covariance
    centered_pc = pc_x - weighted_mean[:, jnp.newaxis, :]
    weighted_cov = jnp.einsum('bij,bik,bi->bjk', centered_pc, centered_pc, normalized_weights)
    
    return weighted_mean, weighted_cov


def covariance_barycenter(cov_matrices, weights=None, max_iter=100, tol=1e-6):
    """
    Compute the Wasserstein barycenter of N covariance matrices.

    Args:
        cov_matrices: Array of shape (N, d, d), where N is the number of matrices, and d is the dimension.
        weights: Optional array of shape (N,) containing the weights of each matrix. If None, uniform weights are used.
        max_iter: Maximum number of iterations for the fixed-point iteration.
        tol: Convergence tolerance.

    Returns:
        The Wasserstein barycenter matrix of shape (d, d).
    """
    N, d, _ = cov_matrices.shape
    if weights is None:
        weights = jnp.ones(N) / N

    # Initialize the barycenter as the weighted average of the covariances
    barycenter = jnp.sum(weights[:, None, None] * cov_matrices, axis=0)

    def fixed_point_iteration(barycenter):
        def update(cov_matrix, barycenter):
            # Compute matrix square root
            sqrt_barycenter = matrix_sqrt(barycenter)
            inv_sqrt_barycenter = jnp.linalg.pinv(sqrt_barycenter)
            transformed_cov = inv_sqrt_barycenter @ cov_matrix @ inv_sqrt_barycenter
            return sqrt_barycenter @ matrix_sqrt(transformed_cov) @ sqrt_barycenter

        barycenter_new = jnp.sum(weights[:, None, None] * jax.vmap(update, in_axes=(0, None))(cov_matrices, barycenter), axis=0)
        return barycenter_new

    for i in range(max_iter):
        barycenter_new = fixed_point_iteration(barycenter)
        if jnp.linalg.norm(barycenter_new - barycenter) < tol:
            break
        barycenter = barycenter_new

    return barycenter

def matrix_sqrt(A):
    """
    Compute the matrix square root using eigendecomposition.
    
    Args:
    A: A symmetric positive definite matrix
    
    Returns:
    The matrix square root of A
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(A)
    eigenvalues = jax.nn.relu(eigenvalues)
    return eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues)) @ eigenvectors.T


def entropic_ot_distance(pc_x, pc_y, eps = 0.1, lse_mode = False, num_iteration = 200): 
    if len(pc_x) == 3:
        pc_x, w_x, _ = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]

    if len(pc_y) == 3:
        pc_y, w_y, _ = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps),
        a = w_x,
        b = w_y,
        lse_mode = lse_mode,
        min_iterations = num_iteration,
        max_iterations = num_iteration)
    return(ot_solve.reg_ot_cost)


def euclidean_distance(pc_x, pc_y): 
    if len(pc_x) == 3:
        pc_x, w_x, _ = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]

    if len(pc_y) == 3:
        pc_y, w_y, _ = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]

    dist_max = jnp.mean(jnp.square(pc_x[:, None, :] - pc_y[None, :, :]), axis = -1)
    weight_mat = jnp.outer(w_x, w_y)

    dist = jnp.sum(dist_max * weight_mat)
    return(dist)

def random_distance(pc_x, pc_y, distance_matrix_func):
    
    if len(pc_x) == 3:
        pc_x, w_x, m_x = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]
        m_x = None

    if len(pc_y) == 3:
        pc_y, w_y, m_y = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]
        m_y = None

    if m_x is not None and m_y is not None:
        pairwise_dist = distance_matrix_func(pc_x, pc_y, m_x, m_y)
    else:
        pairwise_dist = distance_matrix_func(pc_x, pc_y)

    weight_mat = jnp.outer(w_x, w_y)
    dist = jnp.sum(pairwise_dist * weight_mat)
    return(dist)


def matched_distance(pc_x, pc_y, distance_func):
    
    if len(pc_x) == 3:
        pc_x, w_x, m_x = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]
        m_x = None

    if len(pc_y) == 3:
        pc_y, w_y, m_y = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]
        m_y = None

    # assert that w_x and w_y are equal


    if m_x is not None and m_y is not None:
        pairwise_dist = distance_func(pc_x, pc_y, m_x, m_y)
    else:
        pairwise_dist = distance_func(pc_x, pc_y)

    # take only the diagonal elements where both weights are non-zero

    matched_dist = jnp.sum(pairwise_dist * w_x)
    return matched_dist



def chamfer_distance(pc_x, pc_y, distance_matrix_func):
    
    if len(pc_x) == 3:
        pc_x, w_x, m_x = pc_x
    else:
        pc_x, w_x = pc_x[0], pc_x[1]
        m_x = None

    if len(pc_y) == 3:
        pc_y, w_y, m_y = pc_y
    else:
        pc_y, w_y = pc_y[0], pc_y[1]
        m_y = None

    w_x_bool = w_x > 0
    w_y_bool = w_y > 0

    if m_x is not None and m_y is not None:
        pairwise_dist = distance_matrix_func(pc_x, pc_y, m_x, m_y)
    else:
        pairwise_dist = distance_matrix_func(pc_x, pc_y)



    # set pairwise_dist where w_x is zero to infinity

    pairwise_dist += -jnp.log(w_x_bool[:, None] + 1e-6) - jnp.log(w_y_bool[None, :] + 1e-6)
    # use weighted average:

    chamfer_dist = jnp.sum(pairwise_dist.min(axis = 0) * w_y) + jnp.sum(pairwise_dist.min(axis = 1) * w_x)
    return chamfer_dist


def frechet_distance(Nx, Ny):
    """
    Compute the Fréchet distance between two Gaussian distributions.
    
    Args:
    Nx: Mean and covariance of the source distribution (shape: (d,))
    Ny: Mean and covariance of the target distribution (shape: (d,))

    Returns:
    The Fréchet distance between the two distributions
    """


    mu_x, sigma_x = Nx#jnp.mean(pc_x, axis = 0), jnp.cov(pc_x.T)
    mu_y, sigma_y = Ny#jnp.mean(pc_y, axis = 0), jnp.cov(pc_y.T)

    mean_diff_squared = jnp.sum((mu_x - mu_y)**2)
    
    # Compute the sum of the square roots of the eigenvalues of sigma_x @ sigma_y
    sigma_x_sqrt = matrix_sqrt(sigma_x)
    product = sigma_x_sqrt @ sigma_y @ sigma_x_sqrt
    eigenvalues = jnp.linalg.eigvalsh(product)
    trace_term = jnp.sum(jnp.sqrt(jnp.maximum(eigenvalues, 0)))  # Ensure non-negative
    
    # Compute the trace of the sum of covariances
    trace_sum = jnp.trace(sigma_x + sigma_y)
    
    # Compute the Fréchet distance
    return(mean_diff_squared + trace_sum - 2 * trace_term)
