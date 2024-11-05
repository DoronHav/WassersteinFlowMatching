import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore



def lower_tri_to_square(v, n):
    """
    :meta private:
    """
    
    # # Initialize the square matrix with zeros
    # mat = jnp.zeros((n, n))
    
    # # Fill the lower triangular part of the matrix (including the diagonal) with the vector elements
    # mat[jnp.tril_indices(n)] = v
    
    # # Since it's symmetric, copy the lower triangular part to the upper triangular part
    # mat = mat + mat.T - jnp.diag(jnp.diag(mat))
    
    # Create an empty lower triangular matrix

    idx = np.tril_indices(n)
    mat = jnp.zeros((n, n), dtype=v.dtype).at[idx].set(v)
    mat = mat + mat.T - jnp.diag(jnp.diag(mat))
    return mat


def scale_pointclouds(point_clouds, minval = -1, maxval = 1):
    """
    :meta private:
    """
        
    # Get the minimum and maximum values of the point clouds
    pc_minval = jnp.min([jnp.min(pc, axis = 0) for pc in point_clouds])
    pc_maxval = jnp.max([jnp.max(pc, axis = 0) for pc in point_clouds])
    
    scaled_point_clouds = [(pc - pc_minval) / (pc_maxval - pc_minval) * (maxval - minval) + minval for pc in point_clouds]
    
    return scaled_point_clouds

def calc_mean_and_cov(point_clouds, minval = -1, maxval = 1):
    """
    :meta private:
    """

    point_clouds = scale_pointclouds(point_clouds, minval = minval, maxval = maxval)
    # Calculate the mean of the point clouds
    means = np.stack([np.mean(pc) for pc in point_clouds], axis = 0)
    covs = np.stack([np.cov(pc, rowval = False) for pc in point_clouds], axis = 0)

    return means, covs