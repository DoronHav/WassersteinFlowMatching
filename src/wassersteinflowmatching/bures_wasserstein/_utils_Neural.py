import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from flax import linen as nn  # type: ignore

from wassersteinflowmatching.bures_wasserstein.DefaultConfig import DefaultConfig


def fill_triangular(x, d, upper=False):
  """
  Creates a triangular matrix from a single vector of inputs.

  This function is a JAX implementation inspired by
  `tfp.substrates.jax.math.fill_triangular`. It takes a 1D vector
  and builds a 2D square matrix, filling one of the triangular
  portions (either upper or lower) with the elements of `x`. The
  elements are filled row by row.

  Args:
    x: A 1D JAX array containing the elements to fill the matrix with.
       The length of `x` must be a triangular number (e.g., 1, 3, 6, 10...).
    d: An integer representing the dimension of the square matrix to be created.
    upper: A boolean indicating whether to fill the upper or lower triangle.
           - If `True`, an upper-triangular matrix is returned.
           - If `False` (default), a lower-triangular matrix is returned.

  Returns:
    A 2D square JAX array representing the filled triangular matrix.

  Raises:
    ValueError: If the length of `x` is not a triangular number.
  """
  zeros = jnp.zeros((d, d), dtype=x.dtype)

  if upper:
    # For the upper triangle, jnp.triu_indices provides indices in the
    # desired row-by-row order.
    rows, cols = jnp.triu_indices(d)
    return zeros.at[rows, cols].set(x)
  else:
    # For the lower triangle, jnp.tril_indices returns indices in
    # column-by-row order. To get the desired row-by-row filling,
    # we generate the indices manually.
    # This is equivalent to using tf.linalg.fill_triangular.
    rows, cols = jnp.tril_indices(d)
    return zeros.at[rows, cols].set(x)
  
def fill_triangular_inverse(x, upper=False):
  """
  Creates a vector from a single triangular matrix.

  This is the inverse operation of `fill_triangular`. It takes a
  lower or upper triangular matrix and flattens its non-zero elements
  back into a 1D vector.

  Args:
    x: A 2D JAX array representing a lower or upper triangular matrix.
       The function assumes that the input `x` is square.
    upper: A boolean indicating whether `x` is an upper or lower triangular
           matrix.
           - If `True`, `x` is treated as upper-triangular.
           - If `False` (default), `x` is treated as lower-triangular.

  Returns:
    A 1D JAX array representing the flattened triangular elements.
  """
  # Ensure the input is a JAX array
  x = jnp.asarray(x)

  # Check that the input is a 2D square matrix
  if x.ndim != 2 or x.shape[0] != x.shape[1]:
      raise ValueError(f"Input must be a square 2D matrix. Got shape {x.shape}")

  if upper:
    # Get the indices of the upper-triangular elements (including the diagonal)
    indices = jnp.triu_indices_from(x, k=0)
  else:
    # Get the indices of the lower-triangular elements (including the diagonal)
    indices = jnp.tril_indices_from(x, k=0)

  # Use the indices to extract the elements into a 1D array.
  # JAX extracts elements in row-major order, which is the standard
  # way to flatten triangular matrices.
  return x[indices]


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
    def __call__(self, means, cov_tril, t,  labels = None, deterministic = True):
        config = self.config

        embedding_dim = config.embedding_dim
        num_layers = config.num_layers

        freqs = jnp.arange(embedding_dim//2) 

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


            covariance_dot_tril = nn.Dense((space_dim * (space_dim + 1)) // 2,
                                kernel_init=nn.initializers.variance_scaling(1e-3, mode='fan_in', distribution='truncated_normal'), 
                                bias_init=nn.initializers.zeros)(sigma_dot_emb)

        else:

            dot_emb = InputMeanCovarianceNN(config)(means, covariances, t, labels, deterministic)
            dot_emb = FeedForward(config)(inputs = dot_emb, deterministic = deterministic, skip_connection = False, layer_norm = False)


            mean_dot = nn.Dense(space_dim)(dot_emb)
            covariance_dot_tril = nn.Dense((space_dim * (space_dim + 1)) // 2)(dot_emb)

        return mean_dot, covariance_dot_tril


