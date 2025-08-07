import jax # type: ignore
from jax import random   # type: ignore
import jax.numpy as jnp # type: ignore
from jax.scipy.stats import norm, multivariate_normal  # type: ignore

from wassersteinflowmatching.wasserstein.utils_OT import s2_distance

def project_psd(A):
    """Project a matrix onto the positive semidefinite cone."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-6)
    return jnp.dot(eigvecs, jnp.dot(jnp.diag(eigvals), eigvecs.T))

# def sample_gaussian_points(key, cov, n):
#     """Samples n points from a multivariate Gaussian."""
#     dim = cov.shape[0]
#     # Use cholesky decomposition for stable sampling
#     chol = jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(dim)) # Add jitter for stability
#     return jnp.dot(random.normal(key, (n, dim)), chol.T)


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
    
    cov_matrices = jax.vmap(sample_single_wishart)(keys)/df
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
    covariance_barycenter_chol = noise_config.cov_chol_mean
    noise_df_scale = noise_config.noise_df_scale

    K, n, d = size
    df = int(d*noise_df_scale)
    wishart_key, gaussian_key = random.split(key)
    cov_matrices = sample_wishart(wishart_key, df, covariance_barycenter_chol, K) * noise_df_scale
    
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

    # minval = noise_config.minval
    # maxval = noise_config.maxval
    K, n, d = size
    chol_key, gaussian_key = random.split(key, 2)
    
    # Get the mean and std of Cholesky factors
    chol_mean = noise_config.cov_chol_mean
    chol_std = noise_config.cov_chol_std 
    
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
    points = jax.vmap(lambda key, cov: sample_gaussian_points_chol(key, cov, n))(keys, cov_matrices_chol) * noise_config.noise_df_scale
    # points = jnp.clip(points, minval, maxval)
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



def norm_logpdf(x, loc, scale):
    """
    A manual, JIT-friendly implementation of the univariate normal log PDF.
    `loc` is the mean (mu), `scale` is the standard deviation (sigma).
    """
    log_unnormalized = -0.5 * jnp.square((x - loc) / scale)
    log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(scale)
    return log_unnormalized - log_normalization



def multivariate_normal_logpdf(x, mean, cov):
    """
    A manual, JIT-friendly implementation of the multivariate normal log PDF.
    `x` is a batch of data points with shape (num_points, num_dims).
    `mean` is the mean vector with shape (num_dims,).
    `cov` is the covariance matrix with shape (num_dims, num_dims).
    """
    num_dims = x.shape[-1]
    
    # Calculate the log-determinant of the covariance matrix
    sign, log_det_cov = jnp.linalg.slogdet(cov)
    
    # Calculate the Mahalanobis distance term for each point in the batch
    # This term is (x - mean)^T * inv(cov) * (x - mean)
    # We solve this efficiently using jnp.linalg.solve
    x_minus_mean = x - mean
    # We vmap the solver over the batch of points in x
    mahalanobis_sq = jax.vmap(lambda b: b.T @ jnp.linalg.solve(cov, b))(x_minus_mean)

    # Combine the terms for the final log probability for each point
    log_pdf_batch = -0.5 * (num_dims * jnp.log(2. * jnp.pi) + log_det_cov + mahalanobis_sq)
    
    return log_pdf_batch

@staticmethod
def _log_pdf_chol_normal(z, noise_config, with_wasserstein=False, key=random.PRNGKey(0)):

    from wassersteinflowmatching.wasserstein.utils_OT import s2_distance
    """
    Calculates the log probability of a point cloud `z` under the meta-normal prior.
    This uses the approximation: log p(z) ≈ log p(cov(z)|DiagGaussian) + log p(z|N(0, cov(z)))
    The second term is approximated by the negative Wasserstein distance.

    Args:
        z: The input point cloud. Shape: (num_points, num_dims).
        noise_config: A configuration object with `cov_mean` and `cov_std` attributes.
        with_wasserstein: If True, calculates and subtracts the Wasserstein distance.
        key: JAX random key for sampling.

    Returns:
        The calculated log probability.
    """
    # 1. Calculate the empirical covariance of the point cloud z
    empirical_cov = jnp.cov(z, rowvar=False)
    num_dims = z.shape[-1]

    # 2. Calculate the log probability of the empirical covariance under a diagonal Gaussian prior.
    # This corresponds to the first term in the approximation: log p(cov(z)|DiagGaussian)
    log_pdf_cov = norm_logpdf(
        empirical_cov.flatten(),
        loc=noise_config.cov_mean.flatten(),
        scale=noise_config.cov_std.flatten()
    ).sum()

    if with_wasserstein:
        # --- Wasserstein Distance Calculation ---
        # This part approximates the second term: log p(z|N(0, cov(z))).
        cov_no_grad = jax.lax.stop_gradient(empirical_cov)
        wasserstein_key, _ = random.split(key)
        num_points = z.shape[0]
        
        chol = jnp.linalg.cholesky(cov_no_grad + 1e-6 * jnp.eye(num_dims))
        standard_normal_samples = random.normal(wasserstein_key, shape=(num_dims, num_points))
        
        wasserstein_points = (chol @ standard_normal_samples).T
        w2_dist = s2_distance(z, wasserstein_points, 0.01, True, 200) 

        total_log_pdf = log_pdf_cov - w2_dist * z.shape[0]**2
        return total_log_pdf, w2_dist

    return log_pdf_cov, 0

@staticmethod
def _log_pdf_meta_normal(z, noise_config):

    
    """
    Calculates the log probability of a point cloud `z` under the meta-normal prior.
    This uses the approximation: log p(z) ≈ log p(cov(z)|DiagGaussian) + log p(z|N(0, cov(z)))
    """
    # 1. Calculate the empirical covariance of the point cloud z
    empirical_cov = jnp.cov(z, rowvar=False)
    d = z.shape[1]

    # 2. Calculate the parameters for the diagonal Gaussian approximation of the Wishart prior
    df = int(d * noise_config.noise_df_scale)
    V_chol = noise_config.cov_chol_mean
    V = V_chol @ V_chol.T
    scale_factor = noise_config.noise_df_scale

    # Mean of the generated covariance matrices C = W/df * scale_factor, where W ~ Wishart(df, V)
    mean_cov = V * scale_factor

    # Variance of the elements of the generated covariance matrices
    # Var(C_ij) = (scale_factor^2 / df) * (V_ij^2 + V_ii * V_jj)
    V_ii = jnp.diag(V)
    var_cov = (scale_factor**2 / df) * (V**2 + V_ii[:, None] * V_ii[None, :])
    
    # Log probability of the empirical covariance under the diagonal Gaussian approximation
    log_pdf_cov = norm_logpdf(
        empirical_cov.flatten(), 
        loc=mean_cov.flatten(), 
        scale=jnp.sqrt(var_cov.flatten())
    ).sum()
    
    # 3. Calculate the log probability of the points under a Gaussian with that empirical covariance
    # We assume a zero mean for the Gaussian.
    log_pdf_points = multivariate_normal_logpdf(z, mean=jnp.zeros(d), cov=empirical_cov).mean()
    
    return log_pdf_cov #+ log_pdf_points


def log_pdf_student_t(x, noise_config):
    """
    Calculates the log-PDF of a batch of points `x` under a multivariate 
    Student's t-distribution.
    
    Args:
        x: Data points, shape (num_points, num_dims).
        noise_config: A namespace containing `df`, `scale_matrix_chol`, `minval`, `maxval`.
    """
    num_dims = x.shape[-1]
    
    df = noise_config.noise_df_scale
    loc = 0
    scale_matrix_chol = noise_config.cov_chol_mean  # Ch

    # Log-determinant of the scale matrix
    _sign, log_det_scale = jnp.linalg.slogdet(scale_matrix)
    
    # Mahalanobis distance term
    x_minus_loc = x - loc
    mahalanobis = jax.vmap(lambda b: b.T @ jnp.linalg.solve(scale_matrix, b))(x_minus_loc)
    
    # Log-PDF constants
    log_const = (gammaln(df / 2.0) - gammaln((df + num_dims) / 2.0) +
                 0.5 * num_dims * jnp.log(df * jnp.pi) + 0.5 * log_det_scale)
                 
    log_kernel = -0.5 * (df + num_dims) * jnp.log(1.0 + mahalanobis / df)
    
    return -(log_const - log_kernel)

def student_t(size, noise_config, key):
    """
    Generates a batch of point clouds from a Student's t-distribution.
    
    Args:
        size: Tuple of (num_instances, num_points, num_dims).
        noise_config: A namespace containing `df`, `scale_matrix_chol`, `minval`, `maxval`.
        key: JAX random key.
    """
    K, n, d = size
    df = noise_config.noise_df_scale
    scale_matrix_chol = noise_config.cov_chol_mean # Cholesky of Sigma
    
    # 1. Sample from a standard normal
    key_norm, key_gamma = jax.random.split(key)
    z = jax.random.normal(key_norm, shape=(K, n, d))

    # 2. Sample from a Gamma distribution to get the random scaling factor `u`
    # This is equivalent to sampling chi-squared and is more stable.
    u = jax.random.chisquare(key_gamma, df, shape=(K, 1, 1))
    
    # 3. Combine them to get the t-distributed samples
    points = z / jnp.sqrt(u / df)
    
    # 4. Apply the scale matrix (covariance structure) and clip
    scaled_points = points @ scale_matrix_chol.T

    return scaled_points