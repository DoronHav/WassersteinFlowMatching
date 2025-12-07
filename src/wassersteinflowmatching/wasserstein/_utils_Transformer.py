import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import jax.random as random  # type: ignore
from flax import linen as nn  # type: ignore
from typing import Sequence, Optional, Callable

from wassersteinflowmatching.wasserstein.DefaultConfig import WassersteinFlowMatchingConfig

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
    Main attention network, modified to pass conditioning to each EncoderBlock.
    """
    config: WassersteinFlowMatchingConfig

    @nn.compact
    def __call__(self, point_cloud, t, masks = None, labels = None, deterministic = True, dropout_rng=random.key(0)):
        
        config = self.config
        
        embedding_dim = config.embedding_dim
        num_layers = config.num_layers
        space_dim = point_cloud.shape[-1]
        
        # --- 1. Unify Embedding Dimension ---
        # Ensure the embedding dimension is divisible by the number of heads.
        # This will be the single dimension used throughout the transformer.
        embedding_dim = (config.num_heads) * (embedding_dim // (config.num_heads))

        if masks is None:
            masks = jnp.ones((point_cloud.shape[0],point_cloud.shape[1]))

        # --- 2. Create Initial Embeddings ---
        # Embed points, time, and labels into the same dimension, but keep them separate.
        x_emb = nn.Dense(features=embedding_dim)(point_cloud)
            
        freqs = jnp.arange(embedding_dim // 2) * (2.0 * jnp.pi / embedding_dim)
        t_freq = freqs[None, :] * t[:, None]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis = -1)
        t_emb = nn.Dense(features=embedding_dim)(t_four)

        l_emb = None
        if labels is not None:
            if config.discrete_labels:
                label_input = jax.nn.one_hot(labels, config.label_dim)
            else:
                label_input = labels
            l_emb = nn.Dense(features=embedding_dim)(label_input)

        # --- 3. Process through Encoder Stack ---
        # The initial input to the stack is just the point cloud embedding.
        x = x_emb
        # In each layer, re-introduce the time and label embeddings.
        for _ in range(num_layers):
            dropout_rng, layer_dropout_rng = random.split(dropout_rng)
            x = EncoderBlock(config)(
                inputs=x, 
                t_emb=t_emb, 
                l_emb=l_emb, 
                masks=masks, 
                deterministic=deterministic, 
                dropout_rng=layer_dropout_rng
            )   
        
        # --- 4. Final Output Layer ---
        x = nn.Dense(features=space_dim, 
                    kernel_init=nn.initializers.variance_scaling(1e-5, mode='fan_in', distribution='truncated_normal'), 
                    bias_init=nn.initializers.zeros)(x)
        return x

    