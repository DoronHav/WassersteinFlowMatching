import ott # type: ignore
from ott.solvers import linear # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import lax # type: ignore
import jax # type: ignore

def argmax_row_iter(M):
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



def project_to_psd(matrix):
    """
    Project a matrix to the nearest positive semidefinite matrix.
    
    Args:
    matrix: A square matrix
    
    Returns:
    The nearest positive semidefinite matrix to the input matrix
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    eigen_scale = jnp.mean(jnp.abs(eigenvalues))
    return eigenvectors @ jnp.diag(jnp.maximum(eigenvalues, 1e-4 * eigen_scale)) @ eigenvectors.T

def matrix_sqrt(A):
    """
    Compute the matrix square root using eigendecomposition.
    
    Args:
    A: A symmetric positive definite matrix
    
    Returns:
    The matrix square root of A
    """
    A = (A + A.T) / 2  # Ensure symmetry
    eigenvalues, eigenvectors = jnp.linalg.eigh(A)
    eigenvalues = jax.nn.relu(eigenvalues)
    return eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues)) @ eigenvectors.T

def gaussian_monge_map(Nx, Ny):
    """
    Compute the Gaussian Monge map from N(mu_x, sigma_x) to N(mu_y, sigma_y).
    
    Args:
    Nx: Mean and covariance of the source distribution (shape: (d,))
    Ny: Mean and covariance of the target distribution (shape: (d,))

    Returns:
    Parameters for function T(x) that maps points from the source to the target distribution
    """
    
    # Compute A = sigma_y^(1/2) (sigma_y^(1/2) sigma_x sigma_y^(1/2))^(-1/2) sigma_y^(1/2)
    
    mu_x, sigma_x = Nx
    mu_y, sigma_y = Ny

    sigma_y_sqrt = matrix_sqrt(sigma_y)
    inner_sqrt = matrix_sqrt(sigma_y_sqrt @ sigma_x @ sigma_y_sqrt.T)

    
    # Compute A
    A = sigma_y_sqrt @ jnp.linalg.pinv(inner_sqrt) @ sigma_y_sqrt.T
    b = mu_y - A @ mu_x
    # Define the Monge map function
    return A,b

def mccann_interpolation(Nx, T, t):

    
    mu_x, sigma_x = Nx
    A, b = T

    d = mu_x.shape[0]
    Iden = jnp.eye(d)

    mu_t = (1 - t) * mu_x + t * (A @ mu_x + b)
    M = (1 - t) * Iden + t * A
    sigma_t = M @ sigma_x @ M.T
    
    return mu_t, sigma_t

def mccann_derivative(Nx, T, t):
 
    mu_x, sigma_x = Nx
    A, b = T

    d = mu_x.shape[0]
    Iden = jnp.eye(d)

    mu_t_dot = (A-Iden) @ mu_x + b
    sigma_t_dot = (A-Iden) @ sigma_x @ ((1-t) * Iden + t * A).T + ((1-t) * Iden + t * A) @ sigma_x @ (A-Iden).T
    
    return mu_t_dot, sigma_t_dot

def riemann_derivative(Nx, T, t):
 
    mu_x, sigma_x = Nx
    A, b = T

    d = mu_x.shape[0]
    Iden = jnp.eye(d)

    mu_t_dot = (A-Iden) @ mu_x + b
    sigma_t_dot = (A - Iden) @ jnp.linalg.pinv((1-t) * Iden + t * A)
    
    return mu_t_dot, sigma_t_dot



def frechet_distance(Nx, Ny):
    """
    Compute the Fréchet distance between two Gaussian distributions.
    
    Args:
    Nx: Mean and covariance of the source distribution (shape: (d,))
    Ny: Mean and covariance of the target distribution (shape: (d,))

    Returns:
    The Fréchet distance between the two distributions
    """
    mu_x, sigma_x = Nx
    mu_y, sigma_y = Ny

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
    

def euclidean_norm(pred_dot, true_dot, Nt):
    pred_mu_dot, pred_sigma_dot = pred_dot
    true_mu_dot, true_sigma_dot = true_dot

    mean_diff_squared = jnp.sum((pred_mu_dot - true_mu_dot)**2)
    sigma_norm = jnp.sum((pred_sigma_dot - true_sigma_dot)**2)

    return mean_diff_squared, sigma_norm/pred_sigma_dot.shape[-1]

def tangent_norm(pred_dot, true_dot, Nt):
    pred_mu_dot, pred_sigma_dot = pred_dot
    true_mu_dot, true_sigma_dot = true_dot

    mu_t, sigma_t = Nt

    mean_diff_squared = jnp.sum((pred_mu_dot - true_mu_dot)**2)

    sigma_norm = jnp.trace(sigma_t @ (pred_sigma_dot - true_sigma_dot) @ (pred_sigma_dot - true_sigma_dot))

    return mean_diff_squared, sigma_norm

def ot_mat_from_distance(distance_matrix, eps = 0.002, lse_mode = True): 
    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distance_matrix, epsilon = eps, scale_cost = 'max_cost'),
        lse_mode = lse_mode,
        min_iterations = 200,
        max_iterations = 200)
    map_ind = argmax_row_iter(ot_solve.matrix)
    return(map_ind)