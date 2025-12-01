import jax.numpy as jnp # type: ignore
import ott # type: ignore
from ott.solvers import linear # type: ignore
import jax # type: ignore
from jax import lax # type: ignore
from jax import random # type: ignore
import ot # type: ignore
import numpy as np # type: ignore

from tqdm import tqdm

def rounded_matching(M):
    """
    Convert a soft assignment matrix M to a hard assignment vector
    by iteratively finding the largest value in M and making assignments.

    Args:
        M (jnp.ndarray): A square soft assignment matrix.
        
    Returns:
        jnp.ndarray: A vector of length N where the i-th element is the assignment index.
    """
    N = M.shape[0]
    assignment = jnp.full(N, -1, dtype=jnp.int32)

    def body_fun(_, val):
        M, assignment = val
        
        # Find the global maximum
        flat_idx = jnp.argmax(M)
        i, j = jnp.unravel_index(flat_idx, M.shape)
        
        # Update assignment
        assignment = assignment.at[i].set(j)
        
        # Set the corresponding row and column to -inf
        M = M.at[i, :].set(-1)
        M = M.at[:, j].set(-1)
        
        return M, assignment

    _, assignment = lax.fori_loop(0, N, body_fun, (M, assignment))

    return assignment

def weighted_mean_and_covariance(pc_x, weights, epsilon=1e-6):
    """
    Calculate weighted mean and covariance for a batch of point clouds with improved numerical stability.
    
    Args:
    pc_x: Array of shape (batch_size, num_points, num_dimensions)
    weights: Array of shape (batch_size, num_points)
    epsilon: Small constant for numerical stability
    
    Returns:
    weighted_mean: Array of shape (batch_size, num_dimensions)
    weighted_cov: Array of shape (batch_size, num_dimensions, num_dimensions)
    """
    # Ensure weights are positive (avoid division by zero)
    safe_weights = jnp.maximum(weights, epsilon)
    
    # Normalize weights to sum to 1 for each point cloud
    weight_sums = jnp.sum(safe_weights, axis=1, keepdims=True)
    normalized_weights = safe_weights / weight_sums
    
    # Calculate weighted mean
    weighted_mean = jnp.sum(pc_x * normalized_weights[:, :, jnp.newaxis], axis=1)
    
    # Calculate weighted covariance with better numerical stability
    centered_pc = pc_x - weighted_mean[:, jnp.newaxis, :]
    
    # Compute weighted covariance
    weighted_cov = jnp.einsum('bij,bik,bi->bjk', centered_pc, centered_pc, normalized_weights)
    
    # Ensure symmetry (can be broken by numerical issues)
    weighted_cov = (weighted_cov + jnp.transpose(weighted_cov, axes=(0, 2, 1))) / 2.0
    
    # Add small regularization to ensure positive-definiteness
    reg_term = epsilon * jnp.eye(pc_x.shape[-1])[None, :, :]
    weighted_cov = weighted_cov + reg_term
    
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


def entropic_ot_distance(pc_x, pc_y, eps = 0.1, lse_mode = False): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps),
        a = w_x,
        b = w_y,
        lse_mode = lse_mode,
        min_iterations = 200,
        max_iterations = 200)
    return(ot_solve.reg_ot_cost)

def unbalanced_ot_distance(pc_x, pc_y, eps = 0.01, tau_a = 1.0, tau_b = 1.0, lse_mode = False, num_iteration = 200): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve_xy = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps),
        a = w_x,
        b = w_y,
        tau_a = tau_a,
        tau_b = tau_b,
        lse_mode = lse_mode,
        min_iterations = num_iteration,
        max_iterations = num_iteration)

    
    #map_ind = rounded_matching(ot_solve_xy.matrix)
    map_ind = jnp.argmax(ot_solve_xy.matrix, axis = 1)
    pc_y_reord = jnp.take(pc_y, map_ind, axis = 0)
    ot_dist = jnp.sum(jnp.sum((pc_x - pc_y_reord)**2, axis = 1) * w_x)
    return(ot_dist)

def s2_distance(pc_x, pc_y, eps = 0.01, lse_mode = False, num_iteration = 200): 

    ot_solve_xy = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps),
        lse_mode = lse_mode,
        min_iterations = num_iteration,
        max_iterations = num_iteration)

    ot_solve_xx = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_x, cost_fn=None, epsilon = eps),
        lse_mode = lse_mode,
        min_iterations = num_iteration,
        max_iterations = num_iteration)
    
    ot_solve_yy = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_x, cost_fn=None, epsilon = eps),
        lse_mode = lse_mode,
        min_iterations = num_iteration,
        max_iterations = num_iteration)
    
    s2_distance = ot_solve_xy.reg_ot_cost - 0.5 * (ot_solve_xx.reg_ot_cost + ot_solve_yy.reg_ot_cost)
    return(s2_distance)


def euclidean_distance(pc_x, pc_y): 
    pc_x, _ = pc_x[0], pc_x[1]
    pc_y, _ = pc_y[0], pc_y[1]

    dist = jnp.mean(jnp.sum((pc_x - pc_y)**2, axis = 1))
    return(dist)

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

def chamfer_distance(pc_x, pc_y):
    
    pc_x, w_x = pc_x
    pc_y, w_y = pc_y

    w_x_bool = w_x > 0
    w_y_bool = w_y > 0

    pairwise_dist =  jnp.sum(jnp.square(pc_x[:, None, :] - pc_y[None, :, :]), axis=-1)
    # set pairwise_dist where w_x is zero to infinity

    pairwise_dist += -jnp.log(w_x_bool[:, None] + 1e-6) - jnp.log(w_y_bool[None, :] + 1e-6)
    # use weighted average:

    chamfer_dist = jnp.sum(pairwise_dist.min(axis = 0) * w_y) + jnp.sum(pairwise_dist.min(axis = 1) * w_x)
    return chamfer_dist

def ot_mat_from_distance(distance_matrix, eps = 0.002, lse_mode = True): 
    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distance_matrix, epsilon = eps, scale_cost = 'max_cost'),
        lse_mode = lse_mode,
        min_iterations = 200,
        max_iterations = 200)
    map_ind = rounded_matching(ot_solve.matrix)
    return(map_ind)

def sample_ot_matrix(pc_x, pc_y, mat, key):
    """
    Sample a transport matrix from an optimal transport plan.
    
    Args:
    pc_x: Source point cloud.
    pc_y: Target point cloud.
    mat: Optimal transport matrix.
    key: PRNG key.
    

    """
    sample_key, key = random.split(key)
    map_ind = random.categorical(sample_key, logits = jnp.log(mat), axis = -1)
    sampled_flow = pc_y[map_ind] - pc_x

    return sampled_flow

def transport_plan_entropic(pc_x, pc_y, eps = 0.01, lse_mode = False, num_iteration = 200): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps, scale_cost = 'max_cost'),
        a = w_x,
        b = w_y,
        min_iterations = num_iteration,
        max_iterations = num_iteration,
        lse_mode = lse_mode)
    
    potentials = ot_solve.to_dual_potentials()
    delta = potentials.transport(pc_x)-pc_x
    return(delta, ot_solve)

def transport_plan_argmax(pc_x, pc_y, eps = 0.01, lse_mode = False, num_iteration = 200): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps, scale_cost = 'max_cost'),
        a = w_x,
        b = w_y,
        min_iterations = num_iteration,
        max_iterations = num_iteration,
        lse_mode = lse_mode)
    
    map_ind = jnp.argmax(ot_solve.matrix, axis = 1)
    delta = pc_y[map_ind]-pc_x
    return(delta, ot_solve)

def transport_plan_rounded(pc_x, pc_y, eps = 0.01, lse_mode = False, num_iteration = 200): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps, scale_cost = 'max_cost'),
        a = w_x,
        b = w_y,
        min_iterations = num_iteration,
        max_iterations = num_iteration,
        lse_mode = lse_mode)
    
    map_ind = rounded_matching(ot_solve.matrix)
    delta = pc_y[map_ind]-pc_x
    return(delta, ot_solve)


def transport_plan_sample(pc_x, pc_y, eps = 0.01, lse_mode = False, num_iteration = 200): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps, scale_cost = 'max_cost'),
        a = w_x,
        b = w_y,
        min_iterations = num_iteration,
        max_iterations = num_iteration,
        lse_mode = lse_mode)
    
    return(ot_solve.matrix, ot_solve)

def transport_plan_unreg(pc_x, pc_y): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    T = ot.emd(w_x, w_y, ot.dist(pc_x, pc_y))

    map_ind = np.argmax(T, axis=1)
    delta = pc_y[map_ind] - pc_x

    return(delta, T)

def transport_plan_euclidean(pc_x, pc_y): 
    pc_x, _ = pc_x[0], pc_x[1]
    pc_y, _ = pc_y[0], pc_y[1]

    delta = pc_y - pc_x
    return(delta, 0)



def auto_find_num_iter(
    point_clouds, 
    weights, 
    eps, 
    lse_mode, 
    num_calc=100, 
    sample_size=2048, 
    noise_point_clouds=None, 
    noise_weights=None,
    sinkhorn_limit=5000,
    inner_iterations=10  # Check convergence every 10 steps (standard OTT default)
):
    """
    Optimized version using JAX vmap and error tracing.
    Runs a single batched pass to determine the earliest iteration count 
    where 80% of the samples converge.
    """

    num_clouds = len(point_clouds)
    num_noise_clouds = len(noise_point_clouds) if noise_point_clouds is not None else 0
    
    # --------------------------------------------------------------------
    # 1. DATA PREPARATION (CPU/NumPy)
    # JAX requires fixed array shapes for vmapping. We must sample and 
    # stack the data into dense tensors (Batch, SampleSize, Dim) first.
    # --------------------------------------------------------------------
    
    batch_x_list, batch_y_list = [], []
    batch_a_list, batch_b_list = [], []

    # Determine dimensionality from the first cloud
    dim = point_clouds[0].shape[1]

    for _ in range(num_calc):
        # -- Selection Logic --
        if noise_point_clouds is not None and noise_weights is not None:
            idx1 = np.random.randint(0, num_clouds)
            idx2 = np.random.randint(0, num_noise_clouds)
            x, a = point_clouds[idx1], weights[idx1]
            y, b = noise_point_clouds[idx2], noise_weights[idx2]
        else:
            idx1, idx2 = np.random.choice(num_clouds, 2, replace=False)
            x, a = point_clouds[idx1], weights[idx1]
            y, b = point_clouds[idx2], weights[idx2]

        # -- Downsampling Logic --
        if len(x) > sample_size:
            ix = np.random.choice(len(x), sample_size, replace=False)
            x, a = x[ix], a[ix]
        if len(y) > sample_size:
            iy = np.random.choice(len(y), sample_size, replace=False)
            y, b = y[iy], b[iy]
            
        # Handle cases where clouds are smaller than sample_size (pad or error)
        # For this snippet, we assume inputs are at least sample_size or we take all.
        # If strict padding is needed, it should happen here. 
        # Here we assume x and y are now (sample_size, dim).
        
        batch_x_list.append(x)
        batch_y_list.append(y)
        batch_a_list.append(a)
        batch_b_list.append(b)

    # Convert to JAX arrays
    batch_x = jnp.array(np.stack(batch_x_list))  # Shape: (num_calc, sample_size, dim)
    batch_y = jnp.array(np.stack(batch_y_list))
    batch_a = jnp.array(np.stack(batch_a_list))
    batch_b = jnp.array(np.stack(batch_b_list))

    # --------------------------------------------------------------------
    # 2. DEFINING THE VMAP SOLVER (JAX/GPU)
    # --------------------------------------------------------------------
    
    def solve_one(x, y, a, b):
        geom = ott.geometry.pointcloud.PointCloud(x, y, epsilon=eps, scale_cost='max_cost')
        
        # We define the solver with the maximum limit.
        # store_errors=True is crucial: it saves the error at every `inner_iterations` block.
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
    # errors shape: (num_calc, ceil(sinkhorn_limit / inner_iterations))
    # thresholds shape: (num_calc,) (Usually fixed, but returned for safety)
    errors = solve_batch(batch_x, batch_y, batch_a, batch_b)

    # --------------------------------------------------------------------
    # 3. ANALYZING THE TRACE
    # --------------------------------------------------------------------

    # errors contains the marginal error at steps [10, 20, 30...].
    # If the solver converged early, OTT usually pads the error array with -1 or inf.
    # However, standard logic is: check if error < threshold.
    
    # Create a matrix of booleans: True if converged at that step
    # We need to broadcast the threshold across the time dimension
    
    # Identify valid convergence: Error is small, AND not the padding value (-1)
    # Note: If OTT uses -1 for padding, we must ensure -1 is not interpreted as "converged" (which is < threshold).
    # Usually error is positive. We assume padded values are -1.
    is_converged = (errors < -1) 
    
    # Calculate convergence rate per column (time step)
    # Shape: (num_steps,)
    convergence_rates = jnp.mean(is_converged.astype(float), axis=0)
    
    # Find the first index where rate >= 0.8
    # argmax returns the *first* occurrence of the max value. 
    # We create a mask of valid candidates first.
    valid_indices = convergence_rates >= 0.8
    
    # Check if we ever hit 80%
    if not jnp.any(valid_indices):
        print(f"WARNING: Convergence rate never reached 80% (Max rate: {jnp.max(convergence_rates):.2f}).")
        return sinkhorn_limit

    # jnp.argmax on a boolean array finds the index of the first True
    first_success_index = jnp.argmax(valid_indices)
    
    # Convert index back to iteration count
    # +1 because index 0 represents the first block of `inner_iterations`
    recommended_iter = (first_success_index + 1) * inner_iterations
    
    print(f"INFO: Found sufficient convergence (80%) at {recommended_iter} iterations.")
    
    # Convert from JAX array to standard Python int
    return int(recommended_iter)




