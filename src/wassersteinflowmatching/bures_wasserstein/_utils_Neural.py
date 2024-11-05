import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from flax import linen as nn  # type: ignore
import tensorflow_probability.substrates.jax.math as jax_prob # type: ignore

from wassersteinflowmatching.bures_wasserstein.DefaultConfig import DefaultConfig


class FeedForward(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """
    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs, deterministic = True, skip_connection = True, layer_norm = True):
        config = self.config
        
        mlp_hidden_dim = config.mlp_hidden_dim
        #dropout_rate = config.dropout_rate

        x = nn.Dense(features = mlp_hidden_dim)(inputs)
        #x = nn.Dropout(dropout_rate, deterministic=deterministic)(x)
        x = nn.swish(x)
        if(skip_connection):
            x = nn.Dense(inputs.shape[-1])(x) + inputs
        if(layer_norm):
            x = nn.LayerNorm()(x)
        return x

class InputMeanCovarianceNN(nn.Module):

    config: DefaultConfig

    @nn.compact
    def __call__(self, means, covariances, t,  labels = None, deterministic = True):
        config = self.config

        embedding_dim = config.embedding_dim
        num_layers = config.num_layers

        freqs = jnp.arange(embedding_dim//2) 

        cov_tril = jax_prob.fill_triangular_inverse(covariances)

        means_emb = nn.Dense(features = embedding_dim)(means)
        covariances_emb = nn.Dense(features = embedding_dim)(cov_tril)

        t_freq = freqs[None, :] * t[:, None]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis = -1)

        t_emb = nn.Dense(features = embedding_dim)(t_four)

        x = jnp.concatenate([means_emb, covariances_emb, t_emb], axis = -1)

        #means_emb + covariances_emb + t_emb
        
        if(labels is not None):
            l_emb = nn.Dense(features = embedding_dim)(jax.nn.one_hot(labels, config.label_dim))
            x = jnp.concatenate([x, l_emb], axis = -1)

        for _ in range(num_layers):
            x = FeedForward(config)(inputs = x, deterministic = deterministic, skip_connection = True, layer_norm = True)

        return(x)

class BuresWassersteinNN(nn.Module):

    config: DefaultConfig

    @nn.compact
    def __call__(self, means, covariances, t,  labels = None, deterministic = True):
        
        config = self.config
        architecture = config.architecture

        space_dim = means.shape[-1]


        if(architecture == 'separate'):
              
            mean_dot_emb = InputMeanCovarianceNN(config)(means, covariances, t, labels, deterministic)  
            sigma_dot_emb = InputMeanCovarianceNN(config)(means, covariances, t, labels, deterministic)  

            mean_dot = nn.Dense(features=space_dim, 
                        kernel_init=nn.initializers.variance_scaling(1e-3, mode='fan_in', distribution='truncated_normal'), 
                        bias_init=nn.initializers.zeros)(mean_dot_emb)


            tril_vec = nn.Dense((space_dim * (space_dim + 1)) // 2,
                                kernel_init=nn.initializers.variance_scaling(1e-3, mode='fan_in', distribution='truncated_normal'), 
                                bias_init=nn.initializers.zeros)(sigma_dot_emb)

        else:

            dot_emb = InputMeanCovarianceNN(config)(means, covariances, t, labels, deterministic)
            dot_emb = FeedForward(config)(inputs = dot_emb, deterministic = deterministic, skip_connection = False, layer_norm = False)


            mean_dot = nn.Dense(space_dim)(dot_emb)
            tril_vec = nn.Dense((space_dim * (space_dim + 1)) // 2)(dot_emb)

        lower_triangular = jax_prob.fill_triangular(tril_vec)
        covariance_dot = lower_triangular + jnp.triu(lower_triangular.transpose([0,2,1]), k=1)

        return mean_dot, covariance_dot


