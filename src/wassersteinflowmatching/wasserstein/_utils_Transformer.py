import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import jax.random as random  # type: ignore
from flax import linen as nn  # type: ignore


from wassersteinflowmatching.wasserstein.DefaultConfig import DefaultConfig

class FeedForward(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """
    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        mlp_hidden_dim = config.mlp_hidden_dim

        x = nn.Dense(features = mlp_hidden_dim)(inputs)
        x = nn.leaky_relu(x)
        output = nn.Dense(inputs.shape[-1])(x) + inputs
        return output

class EncoderBlock(nn.Module):
    """
    Transformer encoder layer, modified to accept time and label embeddings.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """
    config: DefaultConfig
    
    @nn.compact
    def __call__(self, inputs, t_emb, l_emb, masks, deterministic, dropout_rng = random.key(0)):
        """
        The call signature is updated to accept t_emb and l_emb.
        """
        config = self.config
        num_heads = config.num_heads
        dropout_rate = config.dropout_rate
        
        # --- 1. Create Conditioning Vector ---
        # Combine time and label embeddings. `t_emb` and `l_emb` have shape [batch, dim].
        # We add a `None` dimension to make it [batch, 1, dim] for broadcasting.
        conditioning = t_emb[:, None, :]
        if l_emb is not None:
            conditioning += l_emb[:, None, :]

        # --- 2. Conditioned Attention ---
        # Add the conditioning vector to the input before the attention mechanism.
        # The residual connection is still made from the original, unconditioned input.
        conditioned_inputs = inputs + conditioning
        x = nn.MultiHeadDotProductAttention(
            num_heads = num_heads,
            dropout_rate=dropout_rate,
            deterministic=deterministic
        )(conditioned_inputs, mask = masks[:, None, None, :],  dropout_rng = dropout_rng) + inputs

        x = nn.LayerNorm()(x)
        x = FeedForward(config)(x)
        output = nn.LayerNorm()(x)
        return output

class AttentionNN(nn.Module):
    """
    Main attention network, modified to pass conditioning to each EncoderBlock.
    """
    config: DefaultConfig

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
            x = EncoderBlock(config)(
                inputs=x, 
                t_emb=t_emb, 
                l_emb=l_emb, 
                masks=masks, 
                deterministic=deterministic, 
                dropout_rng=dropout_rng
            )   
        
        # --- 4. Final Output Layer ---
        x = nn.Dense(features=space_dim, 
                    kernel_init=nn.initializers.variance_scaling(1e-5, mode='fan_in', distribution='truncated_normal'), 
                    bias_init=nn.initializers.zeros)(x)
        return x

    