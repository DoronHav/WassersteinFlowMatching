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

# class FeedForward(nn.Module):
#     """Transformer MLP / feed-forward block.

#     Attributes:
#     config: DefaultConfig dataclass containing hyperparameters.
#     out_dim: optionally specify out dimension.
#     """
#     config: DefaultConfig

#     @nn.compact
#     def __call__(self, inputs):
#         mlp_hidden_dim = inputs.shape[-1]

#         x = nn.Dense(features = mlp_hidden_dim)(inputs)
#         x = nn.leaky_relu(x)
#         return x


# class SetNorm(nn.Module):
#     """Set Normalization Module with optional element weights.

#     Attributes:
#         config: DefaultConfig dataclass containing hyperparameters.
#     """

#     config: DefaultConfig
#     @nn.compact
#     def __call__(self, 
#                  x: jnp.ndarray, 
#                  weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
#         """Applies weighted set normalization to the input.

#         Args:
#             x: Input array of shape [B, N, d] where:
#                B = batch size
#                N = set size
#                d = feature dimension
#             weights: Optional weights array of shape [B, N] specifying the weight
#                     of each element in the set. If None, equal weights are used.
#             training: Whether in training mode (currently unused but kept for API consistency)

#         Returns:
#             Normalized input of the same shape as x.
#         """

#         x = jnp.asarray(x)

#         # If no weights provided, use uniform weights
#         if weights is None:
#             weights = jnp.ones(x.shape[:-1])
#         else:
#             weights = jnp.asarray(weights)

#         # Expand weights for broadcasting with features
#         # [B, N] -> [B, N, 1]
#         weights_expanded = jnp.expand_dims(weights, axis=-1)

#         # Normalize weights to sum to 1 for each batch
#         weights_sum = jnp.sum(weights, axis=1, keepdims=True)
#         weights_expanded = weights_expanded / (jnp.maximum(weights_sum[..., None], 1e-6))

#         # Compute weighted mean across both set elements (N) and features (d)
#         # [B, N, d] * [B, N, 1] -> [B, 1, 1]
#         mean = jnp.sum(x * weights_expanded, axis=(1,2), keepdims=True)

#         # Compute weighted variance
#         diff_sq = jnp.square(x - mean)
#         var = jnp.sum(diff_sq * weights_expanded, axis=(1,2), keepdims=True)

#         # Normalize
#         inv = lax.rsqrt(jnp.maximum(var, 1e-6))
#         x = (x - mean) * inv

#         return x
    
# class EncoderBlock(nn.Module):
#     """Transformer encoder layer with Set Normalization.
    
#     Attributes:
#         config: DefaultConfig dataclass containing hyperparameters.
#     """
#     config: DefaultConfig
    
#     @nn.compact
#     def __call__(self, inputs, masks, deterministic, dropout_rng=random.key(0)):
#         config = self.config
#         num_heads = config.num_heads
#         dropout_rate = config.dropout_rate
        
#         # First branch: Apply Set Normalization
#         normalized_inputs = SetNorm(config)(inputs, weights=masks)
        
#         # Apply Multi-head attention
#         attention_output = nn.MultiHeadDotProductAttention(
#             num_heads=num_heads,
#             dropout_rate=dropout_rate,
#             deterministic=deterministic
#         )(normalized_inputs, mask=masks[:, None, None, :], dropout_rng=dropout_rng)
        
#         # First residual connection
#         x = attention_output + inputs
        
#         # Second branch: Apply Set Normalization
#         normalized_x = SetNorm(config)(x, weights=masks)
        
#         # Apply Feed-forward network (weight -> ReLU)
#         ff_output = FeedForward(config)(normalized_x)
        
#         # Second residual connection
#         output = ff_output + x
        
#         return output
    
class EncoderBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """
    config: DefaultConfig

    
    @nn.compact
    def __call__(self, inputs, masks, deterministic, dropout_rng = random.key(0)):

        config = self.config
        num_heads = config.num_heads
        dropout_rate = config.dropout_rate
        # Attention block.
        # x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads = num_heads,
            dropout_rate=dropout_rate,
            deterministic=deterministic
        )(inputs, mask = masks[:, None, None, :],  dropout_rng = dropout_rng) + inputs

        #x = nn.Dropout(rate=config.attention_dropout_rate)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x)
        x = FeedForward(config)(x)
        output = nn.LayerNorm()(x)
        return output

class AttentionNN(nn.Module):

    config: DefaultConfig

    @nn.compact
    def __call__(self, point_cloud, t, masks = None, labels = None, deterministic = True, dropout_rng=random.key(0)):
        
        config = self.config
        
        embedding_dim = config.embedding_dim
        num_layers = config.num_layers

        if(labels is not None):
            embedding_dim =  (3 * config.num_heads) * (embedding_dim // (3 * config.num_heads))
            concat_dim = (embedding_dim // 3)
        else:
            embedding_dim =  (2 * config.num_heads) * (embedding_dim // (2 * config.num_heads))
            concat_dim = (embedding_dim // 2)

        space_dim = point_cloud.shape[-1]

        if masks is None:
            masks = jnp.ones((point_cloud.shape[0],point_cloud.shape[1]))
        x_emb = nn.Dense(features = concat_dim)(point_cloud)
            
        freqs = jnp.arange(embedding_dim) 
        t_freq = freqs[None, :] * t[:, None]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis = -1)

        t_emb = nn.Dense(features = concat_dim)(t_four)

        x = jnp.concatenate([x_emb, 
                             jnp.tile(t_emb[:, None, :], [1, point_cloud.shape[1], 1])], axis = -1)


        if(labels is not None):
            if(config.discrete_labels):
                l_emb = nn.Dense(features = concat_dim)(jax.nn.one_hot(labels, config.label_dim))
            else:
                l_emb = nn.Dense(features = concat_dim)(labels)
            x = jnp.concatenate([x, 
                                 jnp.tile(l_emb[:, None, :], [1, point_cloud.shape[1], 1])], axis = -1)

        
        for _ in range(num_layers):
            x = EncoderBlock(config)(inputs = x, masks = masks, deterministic = deterministic, dropout_rng = dropout_rng)   


        x = nn.Dense(features=space_dim, 
                    kernel_init=nn.initializers.variance_scaling(1e-5, mode='fan_in', distribution='truncated_normal'), 
                    bias_init=nn.initializers.zeros)(x)
        return x



    