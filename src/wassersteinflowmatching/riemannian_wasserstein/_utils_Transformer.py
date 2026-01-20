import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import jax.random as random  # type: ignore
from flax import linen as nn  # type: ignore
from flax.linen.initializers import ones, zeros  # type: ignore
from typing import Sequence, Optional, Callable



class SetLayerNorm(nn.Module):
    epsilon: float = 1e-6
    scale_init: Callable = ones
    bias_init: Callable = zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        
        assert x.ndim == 3, f"Input 'x' must be 3D (B, N, d), but got shape {x.shape}"
        B, N, d = x.shape

        if mask is None:
            mask = jnp.ones((B, N), dtype=jnp.int32)
        else:
            assert mask.ndim == 2, f"Input 'mask' must be 2D (B, N), but got shape {mask.shape}"
            assert x.shape[0] == mask.shape[0], \
                f"Batch dimension mismatch: x {x.shape[0]} vs mask {mask.shape[0]}"
            assert x.shape[1] == mask.shape[1], \
                f"N_points dimension mismatch: x {x.shape[1]} vs mask {mask.shape[1]}"

        mask_expanded = (mask > 0)[..., None].astype(x.dtype)

        num_valid_elements = jnp.sum(mask, axis=-1) * d
        num_valid_elements = num_valid_elements[:, None, None]

        denominator = jnp.maximum(num_valid_elements, 1.0)

        sum_val = jnp.sum(x * mask_expanded, axis=(-2, -1), keepdims=True)
        mean = sum_val / denominator

        sq_diff = jnp.square(x - mean) * mask_expanded
        sum_sq_diff = jnp.sum(sq_diff, axis=(-2, -1), keepdims=True)
        variance = sum_sq_diff / denominator

        x_norm = (x - mean) / jnp.sqrt(variance + self.epsilon)

        scale = self.param('scale', self.scale_init, (d,))
        bias = self.param('bias', self.bias_init, (d,))
        
        y = x_norm * scale + bias

        y_masked = y * mask_expanded

        return y_masked


class FeedForward(nn.Module):
    """
    Transformer MLP / feed-forward block.
    (Pre-norm compatible version)
    """
    config: dict

    @nn.compact
    def __call__(self, inputs, deterministic: bool = True, dropout_rng=None):
        # Use mlp_hidden_dim from your config
        mlp_hidden_dim = self.config.mlp_hidden_dim
        
        x = nn.Dense(features=mlp_hidden_dim)(inputs)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic, rng=dropout_rng)
        x = nn.leaky_relu(x) 
        output = nn.Dense(inputs.shape[-1])(x)
        return output


class EncoderBlock(nn.Module):
    """
    Transformer encoder layer (optionally conditioned) using 
    Pre-Normalization for improved stability in deep networks.
    """

    config: dict

    @nn.compact
    def __call__(
        self, 
        inputs: jnp.ndarray, 
        masks: Optional[jnp.ndarray] = None, 
        dropout_rng: Optional[jnp.ndarray] = None, 
        t_emb: Optional[jnp.ndarray] = None, 
        c_emb: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        
        num_heads = self.config.num_heads
        dropout_rate = self.config.dropout_rate

        # --- 1. Conditioning ---
        # Same logic as before: add time and context embeddings
        # We apply this *before* the first normalization
        
        conditioning = jnp.zeros_like(inputs)
        if t_emb is not None:
            conditioning += t_emb[:, None, :]
        if c_emb is not None:
            conditioning += c_emb[:, None, :]
    
        conditioned_inputs = inputs + conditioning
        attn_mask = masks[:, None, None, :] if masks is not None else None

        attn_rng, ff_rng = jax.random.split(dropout_rng) if dropout_rng is not None else (None, None)
        #normed_inputs = SetLayerNorm()(conditioned_inputs, mask=masks)
        normed_inputs = nn.LayerNorm()(conditioned_inputs)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )(
            normed_inputs, 
            mask=attn_mask, 
            deterministic=deterministic,
            dropout_rng=attn_rng
        )
        
        x = inputs + attn_output
        #normed_x = SetLayerNorm()(x, mask=masks)
        normed_x = nn.LayerNorm()(x)
        ff_output = FeedForward(config=self.config)(normed_x, deterministic=deterministic, dropout_rng=ff_rng)
        output = x + ff_output
        
        return output


class AttentionNN(nn.Module):
    """
    Main attention network for WassersteinFlowMatching.
    
    This version supports selective hypersphere projection for conditioning vectors.
    """

    config: dict

    @nn.compact
    def __call__(
        self,
        point_cloud,
        t,
        masks=None,
        conditioning: Optional[jnp.ndarray] = None,
        is_null_conditioning: Optional[jnp.ndarray] = None, # <-- New argument
        deterministic: bool = True,
        dropout_rng=None,
    ):
        config = self.config
        embedding_dim = config.embedding_dim
        num_layers = config.num_layers
        space_dim = point_cloud.shape[-1]

        embedding_dim = config.num_heads * (embedding_dim // config.num_heads)

        if masks is None:
            masks = jnp.ones((point_cloud.shape[0], point_cloud.shape[1]))

        # 1. Embed point cloud and time (no changes here)
        x_emb = nn.Dense(features=embedding_dim)(point_cloud)

        freqs = jnp.arange(embedding_dim // 2) * (2.0 * jnp.pi / embedding_dim)
        t_freq = freqs[None, :] * t[:, None]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis=-1)
        t_emb = nn.Dense(features=embedding_dim)(t_four)

        # 2. Process conditioning vector with new logic
        c_emb = None
        if conditioning is not None:
            if is_null_conditioning is None:
                is_null_conditioning = jnp.zeros(conditioning.shape[0], dtype=bool)

            # setting as null conditioning if any condition is nan.
            #is_null_conditioning = jnp.isnan(conditioning).any(axis=-1) + is_null_conditioning
            
            if config.normalized_condition:
                # First, pass all conditioning vectors through the dense layer
                c_emb = nn.Dense(features=embedding_dim)(conditioning)

                # Project the embeddings onto the unit hypersphere
                # Use safe norm (eps inside sqrt) to avoid NaN gradients for zero vectors
                sq_norm = jnp.sum(jnp.square(c_emb), axis=-1, keepdims=True)
                norm = jnp.sqrt(sq_norm + 1e-8)
                c_emb = c_emb / norm

                # Selectively set the embedding to zero for samples marked as null
                c_emb = jnp.where(
                    is_null_conditioning[:, None], 
                    jnp.zeros_like(c_emb), 
                    c_emb
                )
            else:
                # Zero out the input conditioning vectors first
                conditioning = jnp.where(
                    is_null_conditioning[:, None],
                    jnp.zeros_like(conditioning),
                    conditioning
                )
                # Then pass through the dense layer
                c_emb = nn.Dense(features=embedding_dim)(conditioning)

        # 3. Apply Transformer blocks (no changes here)
        x = x_emb
        for _ in range(num_layers):
            dropout_rng, key = jax.random.split(dropout_rng) if dropout_rng is not None else (None, None)

            x = EncoderBlock(
                config=config
            )(x, deterministic=deterministic, masks=masks, dropout_rng=dropout_rng, t_emb=t_emb, c_emb=c_emb)

        # 4. Final output layer (no changes here)
        x = nn.Dense(
            features=space_dim,
            kernel_init=nn.initializers.variance_scaling(
                1e-2, mode="fan_in", distribution="truncated_normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        #x = nn.Dense(features=space_dim)(x)
        return x