from jax import random   # type: ignore
import jax.numpy as jnp  # type: ignore 


def gaussian(batch_size, noise_config, key = random.key(0)):

    key_means, key_covs = random.split(key)
    
    # Generate k random means in d dimensions
    means = noise_config.mean_scale * random.normal(key_means, shape=(batch_size, noise_config.dimention))
    
    L = random.normal(key_covs, shape=(batch_size, noise_config.dimention, int(noise_config.dimention  * noise_config.degrees_of_freedom_scale)))
    covariances = noise_config.cov_scale * jnp.matmul(L, jnp.transpose(L, axes=(0, 2, 1)))/int(noise_config.dimention  * noise_config.degrees_of_freedom_scale)
    
    return means, covariances

def sampled_mean_and_cov(batch_size, noise_config, key = random.key(0)):
    noise_means = noise_config.noise_means
    noise_covariances = noise_config.noise_covariances

    if(batch_size>=noise_means.shape[0]):
        batch_size = noise_means.shape[0]
        replace = False
    else:
        replace = True

    ind_key, key = random.split(key)
    
    noise_inds = random.choice(
                key=ind_key,
                a = noise_means.shape[0],
                shape=[batch_size],
                replace=replace)
    
    sampled_means = jnp.take(noise_means, noise_inds, axis=0)
    sampled_covariances = jnp.take(noise_covariances, noise_inds, axis=0)

    return sampled_means, sampled_covariances
