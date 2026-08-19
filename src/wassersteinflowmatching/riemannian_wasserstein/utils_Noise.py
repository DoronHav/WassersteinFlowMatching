import jax # type: ignore
from jax import random   # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np
from tqdm import tqdm





@jax.jit
def _compute_batch_stats(batch_pc, batch_w):
    def body_fn(pc, w):
        normalized_w = w / jnp.sum(w)
        pc_mean = jnp.sum(pc * normalized_w[:, jnp.newaxis], axis=0)
        centered_pc = pc - pc_mean
        pc_cov = jnp.einsum('ij,ik,i->jk', centered_pc, centered_pc, normalized_w)
        cov_chol = jnp.linalg.cholesky(pc_cov + jnp.eye(pc_cov.shape[-1]) * 1e-5)
        return pc_mean, cov_chol
    return jax.vmap(body_fn)(batch_pc, batch_w)

def estimate_euclidean_gaussian_params(point_clouds, weights=None):
    """Estimate Gaussian parameters from data in Euclidean space."""
    point_clouds = np.array(point_clouds)
    K = point_clouds.shape[0]
    
    if weights is None:
        weights = np.ones(point_clouds.shape[:2])
    else:
        weights = np.array(weights)

    means = []
    cov_chols = []
    
    batch_size = 512
    
    for i in tqdm(range(0, K, batch_size), desc="Estimating Gaussian Params"):
        batch_pc = point_clouds[i:i+batch_size]
        batch_w = weights[i:i+batch_size]
        
        # Subsample if needed
        if batch_pc.shape[1] > 2048:
            indices = np.random.choice(batch_pc.shape[1], 2048, replace=False)
            batch_pc = batch_pc[:, indices, :]
            batch_w = batch_w[:, indices]
        
        batch_mean, batch_cov_chol = _compute_batch_stats(batch_pc, batch_w)
        
        means.append(batch_mean)
        cov_chols.append(batch_cov_chol)
    
    means = jnp.concatenate(means, axis=0)
    cov_chols = jnp.concatenate(cov_chols, axis=0)
    
    return {
        'mean': jnp.mean(means, axis=0),
        'cov_chol_mean': jnp.mean(cov_chols, axis=0),
        'cov_chol_std': jnp.std(cov_chols, axis=0),
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
    point_clouds = np.array(point_clouds)
    if weights is None:
        weights = np.ones(point_clouds.shape[:2])
    else:
        weights = np.array(weights)
    
    # Ensure weights sum to 1 for each point cloud in the batch
    normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    # Calculate weighted mean for each point cloud
    point_clouds_mean = np.sum(point_clouds * normalized_weights[:, :, np.newaxis], axis=1)
    
    # Calculate weighted variance for each point cloud (diagonal covariance)
    centered_pc = point_clouds - point_clouds_mean[:, np.newaxis, :]
    # Variance per dimension: sum(w * x^2)
    point_clouds_var = np.sum((centered_pc ** 2) * normalized_weights[:, :, np.newaxis], axis=1)
    point_clouds_std = np.sqrt(point_clouds_var + 1e-5)

    return {
        'mean': jnp.array(np.mean(point_clouds_mean, axis=0)),
        'std_mean': jnp.array(np.mean(point_clouds_std, axis=0)),
        'std_std': jnp.array(np.std(point_clouds_std, axis=0)),
        'noise_df_scale': 1.0,
    }

def estimate_chol_normal_params(point_clouds, weights=None):
    """Same estimator as :func:`estimate_euclidean_gaussian_params`, but with ``noise_df_scale``
    set to 2.0 (the original ``wassersteinflowmatching.wasserstein`` module's default) instead of
    the hardcoded 1.0 -- see :func:`chol_normal_noise`.
    """
    params = estimate_euclidean_gaussian_params(point_clouds, weights)
    params['noise_df_scale'] = 2.0
    return params


def chol_normal_noise(size, noise_config, key, projection_func):
    """Direct port of ``wassersteinflowmatching.wasserstein.utils_Noise.chol_normal``, for
    pinpointing why that module's noise reproduces the WFM paper's reported MERFISH 1-NNA
    (~53-56%) while ``ambient_gaussian`` (this module's default) does not (~90%+ on the same
    data/model/OT config).

    Structurally identical to :func:`euclidean_gaussian_noise` (same per-cloud Cholesky-factor
    perturbation of a mean/std fit across training clouds), but with two differences:

    1. Zero-mean: no ``+ noise_config.mean`` added back (for PCA-centered features this is close
       to a no-op).
    2. ``noise_df_scale`` multiplies the *sampled points* directly (widening each cloud's actual
       spread by 2x linear / 4x variance, the default here), not ``cov_chol_std`` (which only
       widens the *diversity of covariance shapes across clouds*, not any single cloud's spread --
       and is hardcoded to 1.0 in :func:`estimate_euclidean_gaussian_params`, i.e. a no-op there).
    """
    K, n, d = size
    chol_key, gaussian_key = random.split(key)

    chol_mean = noise_config.cov_chol_mean
    chol_std = noise_config.cov_chol_std

    lower_mask = jnp.tril(jnp.ones((d, d)))
    perturbations = random.normal(chol_key, (K, d, d)) * chol_std * lower_mask
    cov_matrices_chol = chol_mean + perturbations

    diag_indices = jnp.diag_indices(d)
    diag_values = jnp.abs(jnp.diagonal(cov_matrices_chol, axis1=1, axis2=2)) + 1e-6
    cov_matrices_chol = cov_matrices_chol.at[:, diag_indices[0], diag_indices[1]].set(diag_values)

    def sample_gaussian_points_chol(key, cov_matrix_chol, n_samples):
        points = random.normal(key, (n_samples, d))
        return points @ cov_matrix_chol.T

    keys = random.split(gaussian_key, K)
    points = jax.vmap(lambda k, cov: sample_gaussian_points_chol(k, cov, n))(keys, cov_matrices_chol)
    points = points * noise_config.noise_df_scale

    projected_points = projection_func(points)
    return projected_points


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


def estimate_degenerate_euclidean_params(point_clouds, weights=None):
    """Estimate Degenerate Gaussian parameters (fixed mean/std) from data in Euclidean space."""
    point_clouds = np.array(point_clouds)
    if weights is None:
        weights = np.ones(point_clouds.shape[:2])
    else:
        weights = np.array(weights)
    
    # Ensure weights sum to 1 for each point cloud in the batch
    normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    # Calculate weighted mean for each point cloud
    point_clouds_mean = np.sum(point_clouds * normalized_weights[:, :, np.newaxis], axis=1)
    
    # Calculate weighted variance for each point cloud (diagonal covariance)
    centered_pc = point_clouds - point_clouds_mean[:, np.newaxis, :]
    # Variance per dimension: sum(w * x^2)
    point_clouds_var = np.sum((centered_pc ** 2) * normalized_weights[:, :, np.newaxis], axis=1)
    point_clouds_std = np.sqrt(point_clouds_var + 1e-5)

    return {
        'mean': jnp.array(np.mean(point_clouds_mean, axis=0)),
        'std': jnp.array(np.mean(point_clouds_std, axis=0)),
    }

def degenerate_euclidean_noise(size, noise_config, key, projection_func):
    """Sample from a fixed Diagonal Gaussian distribution and project to a specified geometry."""
    K, n, d = size
    
    mean = noise_config.mean
    std = noise_config.std

    # Sample points: (K, n, d)
    points = random.normal(key, (K, n, d)) * std + mean
    
    # Project points onto the specified geometry
    projected_points = projection_func(points)
    
    return projected_points


# ##################################################################################################
# Uniform Noise Generation
# ##################################################################################################


def get_noise_functions(noise_type, projection_func=None, geom_utils=None):
    """
    Factory function to get noise generation and parameter estimation functions.

    :param geom_utils: (optional) geometry object; required for geometry-native noise types
        such as 'uniform_mesh', which sample the base distribution directly on the manifold.
    """

    # Geometry-native base distribution: uniform over the mesh surface (no data-driven params).
    if noise_type == 'uniform_mesh':
        if geom_utils is None or not hasattr(geom_utils, 'sample_uniform'):
            raise ValueError("noise type 'uniform_mesh' requires a geometry with sample_uniform (e.g. TriangleMesh)")
        noise_func = lambda size, noise_config, key: geom_utils.sample_uniform(size, key)
        return noise_func, None

    # For uniform noise, there's no parameter estimation from data
    if noise_type == 'ambient_gaussian':
        if projection_func is None:
            raise ValueError("projection_func must be provided for ambient_gaussian on sphere/hyperbolic")
        raw_noise_func = euclidean_gaussian_noise
        noise_func = lambda size, noise_config, key: raw_noise_func(size, noise_config, key, projection_func=projection_func)
        param_estimator = estimate_euclidean_gaussian_params
    elif noise_type == 'chol_normal':
        if projection_func is None:
            raise ValueError("projection_func must be provided for chol_normal")
        raw_noise_func = chol_normal_noise
        noise_func = lambda size, noise_config, key: raw_noise_func(size, noise_config, key, projection_func=projection_func)
        param_estimator = estimate_chol_normal_params
    elif noise_type == 'ambient_diagonal_gaussian':
        if projection_func is None:
            raise ValueError("projection_func must be provided for ambient_diagonal_gaussian on sphere/hyperbolic")
        raw_noise_func = euclidean_diagonal_gaussian_noise
        noise_func = lambda size, noise_config, key: raw_noise_func(size, noise_config, key, projection_func=projection_func)
        param_estimator = estimate_euclidean_diagonal_gaussian_params
    elif noise_type == 'degenerate_euclidean':
        if projection_func is None:
            raise ValueError("projection_func must be provided for degenerate_euclidean")
        raw_noise_func = degenerate_euclidean_noise
        noise_func = lambda size, noise_config, key: raw_noise_func(size, noise_config, key, projection_func=projection_func)
        param_estimator = estimate_degenerate_euclidean_params
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
        
    return noise_func, param_estimator











