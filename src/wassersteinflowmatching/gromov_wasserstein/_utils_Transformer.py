"""
Set Transformer architecture for Gromov-Wasserstein Flow Matching.

The network f_theta(X_t, t) is permutation-equivariant in the point
dimension and conditioned on the scalar time t via sinusoidal embeddings.
"""

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import jax.random as random  # type: ignore
from flax import linen as nn  # type: ignore
from typing import Optional

from wassersteinflowmatching.gromov_wasserstein.DefaultConfig import (  # type: ignore
    GromovWassersteinFlowMatchingConfig,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two-layer MLP with a LeakyReLU activation."""

    config: GromovWassersteinFlowMatchingConfig

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
        dropout_rng=None,
    ) -> jnp.ndarray:
        x = nn.Dense(features=self.config.mlp_hidden_dim)(inputs)
        x = nn.Dropout(rate=self.config.dropout_rate)(
            x, deterministic=deterministic, rng=dropout_rng
        )
        x = nn.leaky_relu(x)
        return nn.Dense(inputs.shape[-1])(x)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder layer with optional time conditioning."""

    config: GromovWassersteinFlowMatchingConfig

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        masks: Optional[jnp.ndarray] = None,
        dropout_rng=None,
        t_emb: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        num_heads = self.config.num_heads
        dropout_rate = self.config.dropout_rate

        # Time conditioning: broadcast over points
        conditioning = jnp.zeros_like(inputs)
        if t_emb is not None:
            conditioning = conditioning + t_emb[:, None, :]

        conditioned = inputs + conditioning
        attn_mask = masks[:, None, None, :] if masks is not None else None

        attn_rng, ff_rng = (
            jax.random.split(dropout_rng)
            if dropout_rng is not None
            else (None, None)
        )

        normed = nn.LayerNorm()(conditioned)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )(
            normed,
            mask=attn_mask,
            deterministic=deterministic,
            dropout_rng=attn_rng,
        )
        x = inputs + attn_out

        normed2 = nn.LayerNorm()(x)
        ff_out = FeedForward(config=self.config)(
            normed2, deterministic=deterministic, dropout_rng=ff_rng
        )
        return x + ff_out


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class SetTransformer(nn.Module):
    """Permutation-equivariant Set Transformer for point-cloud velocity fields.

    Maps ``(X_t, t)`` to a per-point velocity field of the same shape.

    :param config: :class:`GromovWassersteinFlowMatchingConfig` instance.
    """

    config: GromovWassersteinFlowMatchingConfig

    @nn.compact
    def __call__(
        self,
        point_cloud: jnp.ndarray,
        t: jnp.ndarray,
        masks: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        dropout_rng=random.key(0),
    ) -> jnp.ndarray:
        """Forward pass.

        :param point_cloud: ``(B, n, d)`` input point clouds.
        :param t: ``(B,)`` time values in [0, 1].
        :param masks: ``(B, n)`` mask (1 = active point, 0 = padding).
        :param deterministic: Disable dropout when ``True``.
        :param dropout_rng: JAX random key for dropout.
        :return: Velocity field ``(B, n, d)``.
        """
        config = self.config
        space_dim = point_cloud.shape[-1]

        # Align embedding_dim to be divisible by num_heads
        embedding_dim = config.num_heads * (
            config.embedding_dim // config.num_heads
        )

        if masks is None:
            masks = jnp.ones(
                (point_cloud.shape[0], point_cloud.shape[1]), dtype=jnp.float32
            )

        # --- Point embedding ---
        x = nn.Dense(features=embedding_dim)(point_cloud)

        # --- Sinusoidal time embedding ---
        freqs = jnp.arange(embedding_dim // 2) * (2.0 * jnp.pi / embedding_dim)
        t_freq = freqs[None, :] * t[:, None]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis=-1)
        t_emb = nn.Dense(features=embedding_dim)(t_four)

        # --- Encoder stack ---
        for _ in range(config.num_layers):
            dropout_rng, layer_rng = random.split(dropout_rng)
            x = EncoderBlock(config=config)(
                inputs=x,
                t_emb=t_emb,
                masks=masks,
                deterministic=deterministic,
                dropout_rng=layer_rng,
            )

        # --- Output projection ---
        x = nn.Dense(
            features=space_dim,
            kernel_init=nn.initializers.variance_scaling(
                1e-5, mode="fan_in", distribution="truncated_normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        return x
