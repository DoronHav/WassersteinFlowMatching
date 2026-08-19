"""
Noise sampling utilities for Gromov-Wasserstein Flow Matching.

Provides simple source distributions P_0 for sampling noise point clouds.
"""

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import random  # type: ignore


def uniform(size: list, noise_config, key=random.key(0)) -> jnp.ndarray:
    """Sample a batch of point clouds uniformly in [minval, maxval].

    :param size: ``[batch, n_points, d]``.
    :param noise_config: Namespace with ``minval`` and ``maxval`` attributes.
    :param key: JAX random key.
    :return: Array of shape ``size``.
    """
    subkey, _ = random.split(key)
    return random.uniform(
        subkey, shape=size,
        minval=noise_config.minval,
        maxval=noise_config.maxval,
    )


def normal(size: list, noise_config, key=random.key(0)) -> jnp.ndarray:
    """Sample a batch of point clouds from a truncated Normal, rescaled to data range.

    :param size: ``[batch, n_points, d]``.
    :param noise_config: Namespace with ``minval`` and ``maxval`` attributes.
    :param key: JAX random key.
    :return: Array of shape ``size``.
    """
    subkey, _ = random.split(key)
    samples = random.truncated_normal(subkey, lower=-3.0, upper=3.0, shape=size)
    minval = noise_config.minval
    maxval = noise_config.maxval
    return minval + (maxval - minval) * (samples + 3.0) / 6.0


def random_pointclouds(size: list, noise_config, key=random.key(0)) -> tuple:
    """Sample noise point clouds uniformly from a fixed noise dataset.

    Used when the user provides explicit noise point clouds instead of an
    analytical distribution.

    :param size: ``[batch, n_points, d]`` — only ``size[0]`` (batch) is used.
    :param noise_config: Namespace with ``noise_point_clouds`` and
        ``noise_weights`` attributes (arrays pre-loaded from user data).
    :param key: JAX random key.
    :return: ``(sampled_pointclouds, sampled_weights)``.
    """
    ind_key, _ = random.split(key)
    noise_pcs = noise_config.noise_point_clouds
    noise_ws = noise_config.noise_weights
    batch = size[0]
    inds = random.choice(ind_key, noise_pcs.shape[0], shape=(batch,))
    return jnp.take(noise_pcs, inds, axis=0), jnp.take(noise_ws, inds, axis=0)
