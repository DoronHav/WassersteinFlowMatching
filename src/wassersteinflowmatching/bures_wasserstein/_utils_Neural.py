import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from flax import linen as nn  # type: ignore

from wassersteinflowmatching.bures_wasserstein.DefaultConfig import DefaultConfig

def fill_triangular(x, upper=False):
  """
  Creates a triangular matrix from a single vector of inputs.

  This function is a JAX implementation inspired by
  `tfp.substrates.jax.math.fill_triangular`. It takes a 1D vector
  and builds a 2D square matrix, filling one of the triangular
  portions (either upper or lower) with the elements of `x` in a
  clockwise spiral pattern.

  Args:
    x: A 1D JAX array containing the elements to fill the matrix with.
       The length of `x` must be a triangular number (e.g., 1, 3, 6, 10...).
    upper: A boolean indicating whether to fill the upper or lower triangle.
           - If `True`, an upper-triangular matrix is returned.
           - If `False` (default), a lower-triangular matrix is returned.

  Returns:
    A 2D square JAX array representing the filled triangular matrix.
  """
  # --- 1. Calculate matrix dimension 'n' ---
  d = x.shape[-1]
  n = (jnp.sqrt(1. + 8. * d) - 1.) / 2.
  n = jnp.asarray(n, dtype=jnp.int32)

  # --- 2. Build the upper-triangular matrix first ---
  m = n * (n - 1) // 2
  x_tail = x[d - m:]
  xc = jnp.concatenate([x, x_tail[::-1]], axis=0)
  y = jnp.reshape(xc, (n, n))
  m_upper = jnp.triu(y)

  # --- 3. Return the correct matrix based on the 'upper' flag ---
  m_lower = jnp.rot90(m_upper, k=2)
  return jax.lax.cond(
      upper,
      lambda: m_upper,
      lambda: m_lower
  )

def fill_triangular_inverse(x, upper=False):
  """
  Creates a vector from a single triangular matrix.

  This is the inverse operation of `fill_triangular`. It takes a
  lower or upper triangular matrix and flattens its non-zero elements
  back into a 1D vector, following the clockwise spiral pattern.

  Args:
    x: A 2D JAX array representing a lower or upper triangular matrix.
    upper: A boolean indicating whether `x` is an upper or lower triangular
           matrix.
           - If `True`, `x` is treated as upper-triangular.
           - If `False` (default), `x` is treated as lower-triangular.

  Returns:
    A 1D JAX array representing the flattened triangular elements.
  """
  n = x.shape[-1]

  def lower_inverse(mat):
    # For a lower triangular matrix, we read the elements by taking
    # the rows from the bottom-up and reversing them.
    parts = []
    for i in range(n):
      row_idx = n - 1 - i
      # Take the meaningful part of the row
      row = mat[row_idx, :row_idx + 1]
      parts.append(jnp.flip(row))
    return jnp.concatenate(parts)

  def upper_inverse(mat):
    # For an upper triangular matrix, the fill pattern is more complex.
    # We reconstruct the vector by concatenating the specific slices
    # in the order they were filled.
    # 1. The first row.
    # 2. The elements from the remaining rows, starting from the bottom right.
    parts = [mat[0, :]]
    # This loop gathers the remaining elements in the correct spiral order.
    for i in range(1, n):
      row_idx = n - i
      col_idx = n - i
      parts.append(mat[row_idx, col_idx:])
    return jnp.concatenate(parts)

  return jax.lax.cond(
      upper,
      upper_inverse,
      lower_inverse,
      operand=x
  )


# --- Create batched (vmapped) versions of the functions ---
vmapped_fill_triangular = jax.vmap(fill_triangular, in_axes=(0, None), out_axes=0)
vmapped_fill_triangular_inverse = jax.vmap(fill_triangular_inverse, in_axes=(0, None), out_axes=0)

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


