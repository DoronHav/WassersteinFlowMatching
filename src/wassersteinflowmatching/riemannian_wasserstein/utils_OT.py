import jax.numpy as jnp # type: ignore
import ott # type: ignore
from ott.solvers import linear # type: ignore
import jax # type: ignore
from jax import lax # type: ignore
from jax import random # type: ignore



def sample_ot_matrix(mat, key):
    """
    Sample a transport matrix from an optimal transport plan.
    
    Args:
    pc_x: Source point cloud.
    pc_y: Target point cloud.
    mat: Optimal transport matrix.
    key: PRNG key.
    

    """
    sample_key, key = random.split(key)
    map_ind = random.categorical(sample_key, logits = jnp.nan_to_num(jnp.log(mat), nan = -jnp.inf), axis = -1)
    return map_ind

def argmax_row_iter(M, key = random.PRNGKey(0)):
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


def mask_matrix_by_weights(M, row_weights, col_weights):
    # Create masks for zero weights
    row_mask = (row_weights == 0)[:, None]  # Shape (n, 1) for broadcasting
    col_mask = (col_weights == 0)[None, :]  # Shape (1, m) for broadcasting
    
    # Combine masks with OR operation
    combined_mask = jnp.logical_or(row_mask, col_mask)
    
    # Use where to set masked elements to -1
    return jnp.where(combined_mask, -0.5, M)

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


def euclidean_distance(pc_x, pc_y): 
    pc_x, _ = pc_x[0], pc_x[1]
    pc_y, _ = pc_y[0], pc_y[1]

    dist = jnp.mean(jnp.sum((pc_x - pc_y)**2, axis = 1))
    return(dist)

def chamfer_distance(pc_x, pc_y, distance_matrix_func):
    
    pc_x, w_x = pc_x
    pc_y, w_y = pc_y

    w_x_bool = w_x > 0
    w_y_bool = w_y > 0

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

def ot_mat_from_distance(distance_matrix, eps = 0.002, lse_mode = True): 
    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distance_matrix, epsilon = eps, scale_cost = 'max_cost'),
        lse_mode = lse_mode,
        min_iterations = 200,
        max_iterations = 200)
    map_ind = argmax_row_iter(ot_solve.matrix)
    return(map_ind)



def transport_plan(pc_x, pc_y, distance_matrix_func, eps = 0.01, lse_mode = False, num_iteration = 200): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    distmat = distance_matrix_func(pc_x, pc_y)
    
    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distmat, epsilon = eps, scale_cost = 'max_cost'),
        a = w_x,
        b = w_y,
        min_iterations = num_iteration,
        max_iterations = num_iteration,
        lse_mode = lse_mode)
    
    ot_matrix = mask_matrix_by_weights(ot_solve.matrix, w_x, w_y)
    # map_ind = argmax_row_iter(ot_matrix)
    return(ot_matrix, ot_solve)


def transport_plan_euclidean(pc_x, pc_y): 
    matrix = jnp.eye(pc_x.shape[0])
    return(matrix, 0)