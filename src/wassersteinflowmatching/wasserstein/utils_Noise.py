import jax # type: ignore
from jax import random   # type: ignore
import jax.numpy as jnp # type: ignore

def project_psd(A):
    """Project a matrix onto the positive semidefinite cone."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-6)
    return jnp.dot(eigvecs, jnp.dot(jnp.diag(eigvals), eigvecs.T))

def sample_wishart(key, df, V_chol, K):
    """ Sample K covariance matrices from a Wishart distribution.
    
    Args:
        key: PRNG key.
        df: Degrees of freedom.
        V_chol: Cholesky factor of the scale matrix V.
        K: Number of samples.
        
    Returns:
        A tensor of shape (K, dim, dim) where dim is the dimension of the matrix V.
    """
    dim = V_chol.shape[0]
    keys = random.split(key, K)
    
    def sample_single_wishart(key):
        Z = random.normal(key, (df, dim))
        L = Z @ V_chol.T
        return L.T @ L
    
    cov_matrices = jax.vmap(sample_single_wishart)(keys)
    return cov_matrices

def sample_gaussian_points(key, cov_matrix, n):
    """ Sample n points from a zero-mean Gaussian with a given covariance matrix.
    
    Args:
        key: PRNG key.
        cov_matrix: Covariance matrix.
        n: Number of samples.
        
    Returns:
        A tensor of shape (n, dim) where dim is the dimension of the covariance matrix.
    """
    dim = cov_matrix.shape[0]
    L = jnp.linalg.cholesky(project_psd(cov_matrix))
    points = random.normal(key, (n, dim))
    return points @ L.T

def sample_gaussian_points_chol(key, cov_matrix_chol, n):
    """ Sample n points from a zero-mean Gaussian with a given covariance matrix.
    
    Args:
        key: PRNG key.
        cov_matrix: Covariance matrix.
        n: Number of samples.
        
    Returns:
        A tensor of shape (n, dim) where dim is the dimension of the covariance matrix.
    """
    dim = cov_matrix_chol.shape[0]
    points = random.normal(key, (n, dim))
    return points @ cov_matrix_chol.T


def uniform(size, noise_config, key = random.key(0)):
    minval = noise_config.minval
    maxval = noise_config.maxval
    subkey, key = random.split(key)
    noise_samples = random.uniform(subkey, size,
                                    minval = minval, maxval = maxval)
    return(noise_samples)

def normal(size, noise_config, key = random.key(0)):
    minval = noise_config.minval
    maxval = noise_config.maxval
    subkey, key = random.split(key)
    noise_samples = random.truncated_normal(subkey, shape = size, upper = 3, lower = -3)
    noise_samples = minval + (maxval - minval) * (noise_samples + 3) / 6
    return(noise_samples)


# def normal(size, minval, maxval, key = random.key(0)):
#     subkey, key = random.split(key)
#     noise_samples = random.truncated_normal(subkey, shape = size, upper = 3, lower = -3)
#     noise_samples = minval + (maxval - minval) * (noise_samples + 3) / 6
#     return(noise_samples)

def meta_normal(size, noise_config, key):
    """ Sample K covariance matrices from Wishart distribution and n points from each.
    
    Args:
        key: PRNG key.
        V_chol: Cholesky factor of the scale matrix V.
        df: Degrees of freedom.
        K: Number of Wishart samples.
        n: Number of points to sample from Gaussian.
        
    Returns:
        A tuple of (cov_matrices, points), where:
            - cov_matrices is a tensor of shape (K, dim, dim)
            - points is a tensor of shape (K, n, dim)
    """
    # Sample K covariance matrices

    minval = noise_config.minval
    maxval = noise_config.maxval
    covariance_barycenter_chol = noise_config.covariance_barycenter_chol
    noise_df_scale = noise_config.noise_df_scale

    K, n, d = size
    df = int(d*noise_df_scale)
    wishart_key, gaussian_key = random.split(key)
    cov_matrices = sample_wishart(wishart_key, df, covariance_barycenter_chol, K) / df
    
    # Sample n points from Gaussian for each covariance matrix
    keys = random.split(gaussian_key, K)
    points = jax.vmap(lambda key, cov: sample_gaussian_points(key, cov, n))(keys, cov_matrices)
    points = jnp.clip(points, minval, maxval)
    return points

def chol_normal(size, noise_config, key):
    """ Sample K covariance matrices from Wishart distribution and n points from each.
    
    Args:
        key: PRNG key.
        V_chol: Cholesky factor of the scale matrix V.
        df: Degrees of freedom.
        K: Number of Wishart samples.
        n: Number of points to sample from Gaussian.
        
    Returns:
        A tuple of (cov_matrices, points), where:
            - cov_matrices is a tensor of shape (K, dim, dim)
            - points is a tensor of shape (K, n, dim)
    """
    # Sample K covariance matrices

    minval = noise_config.minval
    maxval = noise_config.maxval
    K, n, d = size
    chol_key, gaussian_key = random.split(key, 2)
    
    # Get the mean and std of Cholesky factors
    chol_mean = noise_config.cov_chol_mean
    chol_std = noise_config.cov_chol_std * noise_config.noise_df_scale
    
    # Ensure we're only modifying the lower triangular part
    lower_mask = jnp.tril(jnp.ones((d, d)))
    
    # Generate the perturbations in the lower triangular space
    perturbations = random.normal(chol_key, (K, d, d)) * chol_std * lower_mask
    
    # Use the mean Cholesky factor as the base
    cov_matrices_chol = chol_mean + perturbations
    
    # Force diagonal elements to be positive (required for a valid Cholesky factor)
    diag_indices = jnp.diag_indices(d)
    diag_values = jnp.diagonal(cov_matrices_chol, axis1=1, axis2=2)
    diag_values = jnp.abs(diag_values) + 1e-6  # Ensure positive diagonals
    
    # Update the diagonal elements
    cov_matrices_chol = cov_matrices_chol.at[:, diag_indices[0], diag_indices[1]].set(diag_values)
    
    # Sample n points from Gaussian for each covariance matrix
    keys = random.split(gaussian_key, K)
    points = jax.vmap(lambda key, cov: sample_gaussian_points_chol(key, cov, n))(keys, cov_matrices_chol)
    points = jnp.clip(points, minval, maxval)
    return points

def random_pointclouds(size, noise_config, key):

    noise_pointclouds = noise_config.noise_point_clouds
    noise_weights = noise_config.noise_weights

    ind_key, key = random.split(key)
    
    noise_inds = random.choice(
                key=ind_key,
                a = noise_pointclouds.shape[0],
                shape=[size[0]])
    
    sampled_pointclouds = jnp.take(noise_pointclouds, noise_inds, axis=0)
    sampled_weights = jnp.take(noise_weights, noise_inds, axis=0)

    return sampled_pointclouds, sampled_weights

