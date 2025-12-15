import jax # type: ignore
from jax import random   # type: ignore
import jax.numpy as jnp # type: ignore

# ##################################################################################################
# Hyperspherical Coordinate Noise Generation
# ##################################################################################################

def cartesian_to_hyperspherical(points):
    """Converts Cartesian coordinates on a sphere to hyperspherical coordinates."""
    d = points.shape[-1]
    coords = []
    
    r = jnp.linalg.norm(points, axis=-1, keepdims=True)
    safe_points = points / jnp.where(r > 1e-8, r, 1.0)
    
    remaining_sum_sq = jnp.ones_like(safe_points[..., 0])
    
    for i in range(d - 1):
        angle = jnp.arccos(jnp.clip(safe_points[..., i] / jnp.sqrt(jnp.maximum(remaining_sum_sq, 1e-8)), -1.0, 1.0))
        coords.append(angle)
        remaining_sum_sq -= safe_points[..., i]**2
        
    return jnp.stack(coords, axis=-1)

def hyperspherical_to_cartesian(coords):
    """Converts hyperspherical coordinates to Cartesian coordinates on a sphere using lax.scan."""
    
    # The dimension of the sphere is one more than the number of angles
    d = coords.shape[-1] + 1
    
    # We'll scan over the angles
    # The carry will be the running product of sines
    def scan_body(sin_prod, angle):
        cartesian_coord = sin_prod * jnp.cos(angle)
        next_sin_prod = sin_prod * jnp.sin(angle)
        return next_sin_prod, cartesian_coord

    # Initial product of sines is 1
    initial_sin_prod = jnp.ones_like(coords[..., 0])
    
    # Transpose coords to scan over the angle dimension
    coords_transposed = jnp.moveaxis(coords, -1, 0)
    
    # Run the scan
    final_sin_prod, cartesian_coords_transposed = jax.lax.scan(
        scan_body, initial_sin_prod, coords_transposed
    )
    
    # The last cartesian coordinate is the final product of sines
    # Reshape it to (1, K, n) to match the concatenation axis
    last_coord = final_sin_prod[None, ...]
    
    # Combine the scanned coordinates with the last one
    all_coords_transposed = jnp.concatenate(
        [cartesian_coords_transposed, last_coord], axis=0
    )
    
    # Transpose back to the original dimension order
    points = jnp.moveaxis(all_coords_transposed, 0, -1)
    
    return points


def spherical_hyperspherical_gaussian(size, noise_config, key):
    """Sample from a Gaussian distribution in hyperspherical coordinates."""
    K, n, d = size
    d_minus_1 = d - 1
    
    wishart_key, gaussian_key, means_key = random.split(key, 3)

    # Sample K means
    means = random.normal(means_key, (K, d_minus_1)) * noise_config.mean_std + noise_config.mean_mean

    # Sample K covariance matrices' cholesky decompositions
    cov_matrices_chol = (random.normal(wishart_key, (K, d_minus_1, d_minus_1)) * noise_config.cov_chol_std + noise_config.cov_chol_mean)

    def sample_gaussian_points_chol(key, cov_matrix_chol, n_samples):
        points = random.normal(key, (n_samples, d_minus_1))
        return points @ cov_matrix_chol.T

    # Sample n points from Gaussian for each covariance matrix
    keys = random.split(gaussian_key, K)
    hyperspherical_points = jax.vmap(lambda k, cov, mean: sample_gaussian_points_chol(k, cov, n) + mean)(keys, cov_matrices_chol, means)
    
    # 3. Convert back to Cartesian coordinates
    cartesian_points = hyperspherical_to_cartesian(hyperspherical_points)
    
    return cartesian_points

def estimate_hyperspherical_gaussian_params(point_clouds, weights=None):
    """Estimate hyperspherical Gaussian parameters from spherical data."""
    if weights is None:
        weights = jnp.ones(point_clouds.shape[:2])
    
    # Convert all point clouds to hyperspherical coordinates
    hyperspherical_coords = cartesian_to_hyperspherical(point_clouds)
    
    # Ensure weights sum to 1 for each point cloud in the batch
    normalized_weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    
    # Calculate weighted mean for each point cloud
    point_clouds_mean = jnp.sum(hyperspherical_coords * normalized_weights[:, :, jnp.newaxis], axis=1)
    
    # Calculate weighted covariance for each point cloud
    centered_pc = hyperspherical_coords - point_clouds_mean[:, jnp.newaxis, :]
    point_clouds_cov = jnp.einsum('bij,bik,bi->bjk', centered_pc, centered_pc, normalized_weights)

    cov_chol = jax.vmap(jnp.linalg.cholesky)(point_clouds_cov + jnp.eye(point_clouds_cov.shape[-1]) * 1e-5)
    
    return {
        'mean_mean': jnp.mean(point_clouds_mean, axis=0),
        'mean_std': jnp.std(point_clouds_mean, axis=0),
        'cov_chol_mean': jnp.mean(cov_chol, axis=0),
        'cov_chol_std': jnp.std(cov_chol, axis=0),
    }

# ##################################################################################################
# Hyperbolic Gaussian Noise Generation
# ##################################################################################################

def estimate_hyperbolic_gaussian_params(point_clouds, weights=None):
    """Estimate Gaussian parameters from data in the Poincaré ball."""
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
        'mean_mean': jnp.mean(point_clouds_mean, axis=0),
        'mean_std': jnp.std(point_clouds_mean, axis=0),
        'cov_chol_mean': jnp.mean(cov_chol, axis=0),
        'cov_chol_std': jnp.std(cov_chol, axis=0),
    }

def hyperbolic_gaussian_noise(size, noise_config, key):
    """Sample from a Gaussian distribution and project to the Poincaré ball."""
    K, n, d = size
    
    wishart_key, gaussian_key, means_key = random.split(key, 3)

    # Sample K means
    means = random.normal(means_key, (K, d)) * noise_config.mean_std + noise_config.mean_mean

    # Sample K covariance matrices' cholesky decompositions
    cov_matrices_chol = (random.normal(wishart_key, (K, d, d)) * noise_config.cov_chol_std + noise_config.cov_chol_mean)

    def sample_gaussian_points_chol(key, cov_matrix_chol, n_samples):
        points = random.normal(key, (n_samples, d))
        return points @ cov_matrix_chol.T

    # Sample n points from Gaussian for each covariance matrix
    keys = random.split(gaussian_key, K)
    points = jax.vmap(lambda k, cov, mean: sample_gaussian_points_chol(k, cov, n) + mean)(keys, cov_matrices_chol, means)
    
    # 3. Project points back into the Poincaré ball
    norms = jnp.linalg.norm(points, axis=-1, keepdims=True)
    # Project points with norm > 1 back to the boundary
    projected_points = jnp.where(norms > 1, points / norms, points)
    
    return projected_points

# ##################################################################################################
# Euclidean Gaussian Noise Generation (with projection)
# ##################################################################################################

def estimate_euclidean_gaussian_params(point_clouds, weights=None):
    """Estimate Gaussian parameters from data in Euclidean space."""
    print("Estimating Euclidean Gaussian parameters...")

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

# ##################################################################################################
# Torus Gaussian Noise Generation
# ##################################################################################################

def estimate_torus_gaussian_params(point_clouds, weights=None):
    """Estimate Gaussian parameters from data on the flat torus."""
    if weights is None:
        weights = jnp.ones(point_clouds.shape[:2])
        
    # Ensure weights sum to 1 for each point cloud in the batch
    normalized_weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    
    # Calculate weighted mean for each point cloud
    # For circular data, this is a simplification.
    point_clouds_mean = jnp.sum(point_clouds * normalized_weights[:, :, jnp.newaxis], axis=1)
    
    # Calculate weighted covariance for each point cloud
    centered_pc = point_clouds - point_clouds_mean[:, jnp.newaxis, :]
    point_clouds_cov = jnp.einsum('bij,bik,bi->bjk', centered_pc, centered_pc, normalized_weights)

    cov_chol = jax.vmap(jnp.linalg.cholesky)(point_clouds_cov + jnp.eye(point_clouds_cov.shape[-1]) * 1e-5)
    
    return {
        'mean_mean': jnp.mean(point_clouds_mean, axis=0),
        'mean_std': jnp.std(point_clouds_mean, axis=0),
        'cov_chol_mean': jnp.mean(cov_chol, axis=0),
        'cov_chol_std': jnp.std(cov_chol, axis=0),
    }

def torus_gaussian_noise(size, noise_config, key):
    """Sample from a Gaussian distribution and wrap onto the torus."""
    K, n, d = size
    
    wishart_key, gaussian_key, means_key = random.split(key, 3)

    # Sample K means
    means = random.normal(means_key, (K, d)) * noise_config.mean_std + noise_config.mean_mean

    # Sample K covariance matrices' cholesky decompositions
    cov_matrices_chol = (random.normal(wishart_key, (K, d, d)) * noise_config.cov_chol_std + noise_config.cov_chol_mean)

    def sample_gaussian_points_chol(key, cov_matrix_chol, n_samples):
        points = random.normal(key, (n_samples, d))
        return points @ cov_matrix_chol.T

    # Sample n points from Gaussian for each covariance matrix
    keys = random.split(gaussian_key, K)
    points = jax.vmap(lambda k, cov, mean: sample_gaussian_points_chol(k, cov, n) + mean)(keys, cov_matrices_chol, means)
    
    # 3. Wrap points to the [0, 2*pi) torus
    wrapped_points = points % (2 * jnp.pi)
    
    return wrapped_points

# ##################################################################################################
# Uniform Noise Generation
# ##################################################################################################

def sphere_uniform_noise(size, key, **kwargs):
    """Sample uniformly from the sphere."""
    K, n, d = size
    points = random.normal(key, (K, n, d))
    return points / jnp.linalg.norm(points, axis=-1, keepdims=True)

def torus_uniform_noise(size, key, **kwargs):
    """Sample uniformly from the flat torus [0, 2*pi)^d."""
    K, n, d = size
    return random.uniform(key, (K, n, d), minval=0, maxval=2 * jnp.pi)

def hyperbolic_uniform_noise(size, key, **kwargs):
    """Sample uniformly from the Poincaré ball via rejection sampling."""
    K, n, d = size
    
    num_samples_to_generate = K * n
    
    # Oversample to account for rejections. The volume ratio of a d-ball to a d-cube
    # decreases with d, so we might need a larger factor for high dimensions.
    oversampling_factor = 2.0 
    num_to_sample = int(num_samples_to_generate * oversampling_factor)
    
    # Generate uniform samples in the ambient cube [-1, 1]^d
    # We generate more samples than needed and then filter.
    
    accepted_samples = jnp.zeros((0, d))
    needed = num_samples_to_generate
    
    def cond_fun(state):
        _, _, key, needed = state
        return needed > 0

    def body_fun(state):
        accepted_samples, i, key, needed = state
        subkey, key = random.split(key)
        
        # Generate a batch of samples
        samples = random.uniform(subkey, (num_to_sample, d), minval=-1.0, maxval=1.0)
        
        # Filter samples that are inside the unit ball
        norms = jnp.linalg.norm(samples, axis=-1)
        valid_samples = samples[norms <= 1.0]
        
        # Append valid samples
        new_accepted = jnp.concatenate([accepted_samples, valid_samples])
        
        return new_accepted, i + 1, key, needed - len(valid_samples)

    # Loop until we have enough samples
    final_samples, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, (accepted_samples, 0, key, needed))
    
    # Take the required number of samples and reshape
    points = final_samples[:num_samples_to_generate].reshape(K, n, d)
    
    return points

# ##################################################################################################
# Hyperbolic Intrinsic Gaussian Noise
# ##################################################################################################

def poincare_exp_map(p, v):
    """Exponential map on the Poincaré ball at point p for vector v."""
    norm_v = jnp.linalg.norm(v)
    norm_p_sq = jnp.sum(p**2)
    
    lambda_p = 2 / (1 - norm_p_sq + 1e-6)
    
    # Tangent vector in tangent space of p, scaled
    u = jnp.tanh(lambda_p * norm_v / 2) * v / (norm_v + 1e-6)
    
    # Mobius addition
    norm_u_sq = jnp.sum(u**2)
    
    # Numerator: (1 + 2<p,u> + ||u||^2)p + (1-||p||^2)u
    num = (1 + 2 * jnp.dot(p, u) + norm_u_sq) * p + (1 - norm_p_sq) * u
    
    # Denominator: 1 + 2<p,u> + ||p||^2 ||u||^2
    den = 1 + 2 * jnp.dot(p, u) + norm_p_sq * norm_u_sq
    
    return num / (den + 1e-6)

def hyperbolic_intrinsic_gaussian(size, noise_config, key):
    """Sample from a Gaussian in the tangent space and map to Poincaré ball."""
    K, n, d = size
    
    mu_key, sig_key, sample_key = random.split(key, 3)
    
    # 1. Sample K means (in the ball) and covariance matrices (in tangent space at origin)
    # Sample means uniformly for simplicity
    mean_centers = random.normal(mu_key, (K, d))
    mean_centers = mean_centers / jnp.linalg.norm(mean_centers, axis=-1, keepdims=True) * \
                   random.uniform(mu_key, (K, 1), minval=0.0, maxval=0.8) # Keep means away from boundary

    chol_sigs = random.normal(sig_key, (K, d, d)) * noise_config.std_sig + noise_config.mu_sig
    sigmas = jax.vmap(lambda L: jnp.dot(L, L.T))(chol_sigs)
    
    # 2. Sample n tangent vectors for each of the K distributions
    def sample_single(mu, sigma, k):
        tangent_vectors = random.multivariate_normal(k, jnp.zeros(d), sigma, (n,))
        # Map tangent vectors at mu to the ball
        return jax.vmap(poincare_exp_map, in_axes=(None, 0))(mu, tangent_vectors)

    keys = random.split(sample_key, K)
    points = jax.vmap(sample_single)(mean_centers, sigmas, keys)
    
    return points

def estimate_hyperbolic_intrinsic_gaussian_params(point_clouds, weights=None):
    """
    Estimate parameters for intrinsic hyperbolic Gaussian.
    This is a simplification: it estimates covariance in the ambient space,
    which is an approximation of covariance in tangent spaces.
    """
    # For simplicity, we use the same estimation as the ambient Gaussian.
    # A more accurate implementation would involve mapping points to a tangent space,
    # computing stats there, which is more complex.
    return estimate_hyperbolic_gaussian_params(point_clouds, weights)

# ##################################################################################################
# Noise Factory
# ##################################################################################################

def get_noise_functions(geom, noise_type, projection_func=None):
    """
    Factory function to get noise generation and parameter estimation functions.
    """
    
    # For uniform noise, there's no parameter estimation from data
    if noise_type == 'uniform':
        if geom == 'sphere':
            return sphere_uniform_noise, None
        elif geom == 'torus':
            return torus_uniform_noise, None
        elif geom == 'hyperbolic':
            return hyperbolic_uniform_noise, None
        else:
            raise ValueError(f"Unsupported geometry for uniform noise: {geom}")

    # For Gaussian noise types
    if noise_type == 'intrinsic_gaussian':
        if geom == 'sphere':
            noise_func = spherical_hyperspherical_gaussian
            param_estimator = estimate_hyperspherical_gaussian_params
        elif geom == 'torus':
            # Intrinsic and ambient are the same for the flat torus
            noise_func = torus_gaussian_noise
            param_estimator = estimate_torus_gaussian_params
        elif geom == 'hyperbolic':
            noise_func = hyperbolic_intrinsic_gaussian
            param_estimator = estimate_hyperbolic_intrinsic_gaussian_params
        else:
            raise ValueError(f"Unsupported geometry for intrinsic Gaussian: {geom}")
            
    elif noise_type == 'ambient_gaussian':
        if geom in ['sphere', 'hyperbolic']:
            if projection_func is None:
                raise ValueError("projection_func must be provided for ambient_gaussian on sphere/hyperbolic")
            raw_noise_func = euclidean_gaussian_noise
            noise_func = lambda size, noise_config, key: raw_noise_func(size, noise_config, key, projection_func=projection_func)
            param_estimator = estimate_euclidean_gaussian_params
        elif geom == 'torus':
            # Ambient is the same as intrinsic for the flat torus
            noise_func = torus_gaussian_noise
            param_estimator = estimate_torus_gaussian_params
        else:
            raise ValueError(f"Unsupported geometry for ambient Gaussian: {geom}")
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
        
    return noise_func, param_estimator











