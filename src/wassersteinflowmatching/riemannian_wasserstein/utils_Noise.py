import jax # type: ignore
from jax import random   # type: ignore
import jax.numpy as jnp # type: ignore





def estimate_euclidean_gaussian_params(point_clouds, weights=None):
    """Estimate Gaussian parameters from data in Euclidean space."""
    if weights is None:
        weights = jnp.ones(point_clouds.shape[:2])
    
    # Ensure weights sum to 1 for each point cloud in the batch
    normalized_weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    
    # Calculate weighted mean for each point cloud
    point_clouds_mean = jnp.sum(point_clouds * normalized_weights[:, :, jnp.newaxis], axis=1)
    
    # Calculate weighted covariance for each point cloud
    centered_pc = point_clouds - point_clouds_mean[:, jnp.newaxis, :]
    point_clouds_cov = jnp.einsum('bij,bik,bi->bjk', centered_pc, centered_pc, normalized_weights)

    cov_chol = jax.vmap(jnp.linalg.cholesky)(point_clouds_cov + jnp.eye(point_clouds_cov.shape[-1]) * 1e-5)
    
    return {
        'mean': jnp.mean(point_clouds_mean, axis=0),
        'cov_chol_mean': jnp.mean(cov_chol, axis=0),
        'cov_chol_std': jnp.std(cov_chol, axis=0),
        'noise_df_scale': 1.0,
    }

def euclidean_gaussian_noise(size, noise_config, key, projection_func):
    """Sample from a Gaussian distribution and project to a specified geometry."""
    K, n, d = size
    
    wishart_key, gaussian_key = random.split(key)

    cov_chol_mean = noise_config.cov_chol_mean
    cov_chol_std = noise_config.cov_chol_std * noise_config.noise_df_scale

    # Sample K covariance matrices' cholesky decompositions
    cov_matrices_chol = (random.normal(wishart_key, (K, d, d)) * cov_chol_std + cov_chol_mean)

    def sample_gaussian_points_chol(key, cov_matrix_chol, n_samples):
        points = random.normal(key, (n_samples, d))
        return points @ cov_matrix_chol.T

    # Sample n points from Gaussian for each covariance matrix
    keys = random.split(gaussian_key, K)
    points = jax.vmap(lambda k, cov: sample_gaussian_points_chol(k, cov, n))(keys, cov_matrices_chol) + noise_config.mean
    
    # Project points onto the specified geometry
    projected_points = projection_func(points)
    
    return projected_points


def estimate_euclidean_diagonal_gaussian_params(point_clouds, weights=None):
    """Estimate Diagonal Gaussian parameters from data in Euclidean space."""
    if weights is None:
        weights = jnp.ones(point_clouds.shape[:2])
    
    # Ensure weights sum to 1 for each point cloud in the batch
    normalized_weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    
    # Calculate weighted mean for each point cloud
    point_clouds_mean = jnp.sum(point_clouds * normalized_weights[:, :, jnp.newaxis], axis=1)
    
    # Calculate weighted variance for each point cloud (diagonal covariance)
    centered_pc = point_clouds - point_clouds_mean[:, jnp.newaxis, :]
    # Variance per dimension: sum(w * x^2)
    point_clouds_var = jnp.sum((centered_pc ** 2) * normalized_weights[:, :, jnp.newaxis], axis=1)
    point_clouds_std = jnp.sqrt(point_clouds_var + 1e-5)

    return {
        'mean': jnp.mean(point_clouds_mean, axis=0),
        'std_mean': jnp.mean(point_clouds_std, axis=0),
        'std_std': jnp.std(point_clouds_std, axis=0),
        'noise_df_scale': 1.0,
    }

def euclidean_diagonal_gaussian_noise(size, noise_config, key, projection_func):
    """Sample from a Diagonal Gaussian distribution and project to a specified geometry."""
    K, n, d = size
    
    std_key, gaussian_key = random.split(key)

    std_mean = noise_config.std_mean
    std_std = noise_config.std_std * noise_config.noise_df_scale

    # Sample K std vectors
    batch_stds = jnp.abs(random.normal(std_key, (K, d)) * std_std + std_mean)

    def sample_diag_gaussian_points(key, stds, n_samples):
        points = random.normal(key, (n_samples, d))
        return points * stds

    # Sample n points from Gaussian for each std vector
    keys = random.split(gaussian_key, K)
    points = jax.vmap(lambda k, s: sample_diag_gaussian_points(k, s, n))(keys, batch_stds) + noise_config.mean
    
    # Project points onto the specified geometry
    projected_points = projection_func(points)
    
    return projected_points


# ##################################################################################################
# Uniform Noise Generation
# ##################################################################################################


def get_noise_functions(noise_type, projection_func=None):
    """
    Factory function to get noise generation and parameter estimation functions.
    """
    
    # For uniform noise, there's no parameter estimation from data
    if noise_type == 'ambient_gaussian':
        if projection_func is None:
            raise ValueError("projection_func must be provided for ambient_gaussian on sphere/hyperbolic")
        raw_noise_func = euclidean_gaussian_noise
        noise_func = lambda size, noise_config, key: raw_noise_func(size, noise_config, key, projection_func=projection_func)
        param_estimator = estimate_euclidean_gaussian_params
    elif noise_type == 'ambient_diagonal_gaussian':
        if projection_func is None:
            raise ValueError("projection_func must be provided for ambient_diagonal_gaussian on sphere/hyperbolic")
        raw_noise_func = euclidean_diagonal_gaussian_noise
        noise_func = lambda size, noise_config, key: raw_noise_func(size, noise_config, key, projection_func=projection_func)
        param_estimator = estimate_euclidean_diagonal_gaussian_params
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
        
    return noise_func, param_estimator











