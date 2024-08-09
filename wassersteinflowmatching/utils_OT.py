import jax.numpy as jnp # type: ignore
import ott # type: ignore
from ott.solvers import linear # type: ignore
import jax # type: ignore

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


def ot_mat_from_distance(distance_matrix, eps = 0.1, lse_mode = False): 
    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distance_matrix, epsilon = eps, scale_cost = 'mean'),
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    return(ot_solve.matrix)

def entropic_ot_distance(pc_x, pc_y, eps = 0.1, lse_mode = False): 

    
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps, scale_cost = 'mean'),
        a = w_x,
        b = w_y,
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    return(ot_solve.reg_ot_cost)

def frechet_distance(Nx, Ny, eps = 0.1, lse_mode = False):
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

def transport_plan_entropic(pc_x, pc_y, eps = 0.001, lse_mode = True): 
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps),
        a = w_x,
        b = w_y,
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    
    potentials = ot_solve.to_dual_potentials()
    delta = potentials.transport(pc_x)-pc_x
    return(delta)

def transport_plan_exact(pc_x, pc_y, eps = 0.001, lse_mode = True):
    pc_x, w_x = pc_x[0], pc_x[1]
    pc_y, w_y = pc_y[0], pc_y[1]

    ot_solve = linear.solve(
        ott.geometry.pointcloud.PointCloud(pc_x, pc_y, cost_fn=None, epsilon = eps),
        a = w_x,
        b = w_y,
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    
    map_ind = jnp.argmax(ot_solve.matrix, axis = 1)
    delta = pc_y[map_ind]-pc_x
    return(delta)