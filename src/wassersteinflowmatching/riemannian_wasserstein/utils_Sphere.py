import jax.numpy as jnp # type: ignore

def distance(P0, P1):
    # Normalize P0 and P1 to ensure they are on the sphere S2
    P0 = jnp.nan_to_num(P0 / jnp.linalg.norm(P0), nan = 1/jnp.sqrt(P0.shape[-1]))
    P1 = jnp.nan_to_num(P1 / jnp.linalg.norm(P1), nan = 1/jnp.sqrt(P1.shape[-1]))
    
    # Compute the dot product between the two points
    dot_product = jnp.dot(P0, P1)
    
    # Clip the dot product to avoid numerical issues with arccos
    dot_product = jnp.clip(dot_product, -1.0, 1.0)
    
    # Compute the great circle distance (angular distance)
    return jnp.arccos(dot_product)

def distance_matrix(P0, P1):
    # Normalize the points to ensure they are on the sphere S2
    P0 = jnp.nan_to_num(P0 / jnp.linalg.norm(P0, axis=1, keepdims=True), nan = 1/jnp.sqrt(P0.shape[-1]))
    P1 = jnp.nan_to_num(P1 / jnp.linalg.norm(P1, axis=1, keepdims=True), nan = 1/jnp.sqrt(P1.shape[-1]))
    
    # Compute the dot product matrix
    dot_product_matrix = P0 @ P1.T
    
    # Clip the dot product matrix to avoid numerical issues with arccos
    dot_product_matrix = jnp.clip(dot_product_matrix, -1.0, 1.0)
    
    # Compute the great circle distance matrix
    return jnp.arccos(dot_product_matrix)

def interpolant(P0, P1, t):
    # Normalize P0 and P1 to ensure they are on the sphere S2
    P0 = jnp.nan_to_num(P0 / jnp.linalg.norm(P0), nan = 1/jnp.sqrt(P0.shape[-1]))
    P1 = jnp.nan_to_num(P1 / jnp.linalg.norm(P1), nan = 1/jnp.sqrt(P1.shape[-1]))
    
    # Compute the cosine of the angle between P0 and P1
    cos_theta = jnp.dot(P0, P1)
    
    # Clip cos_theta to avoid numerical issues with acos
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle theta
    theta = jnp.arccos(cos_theta)
    
    # Compute the sin of theta
    sin_theta = jnp.sin(theta)
    
    # Use jnp.where to smoothly handle the case where sin_theta is small
    a = jnp.where(sin_theta < 1e-6, 1.0 - t, jnp.sin((1 - t) * theta) / sin_theta)
    b = jnp.where(sin_theta < 1e-6, t, jnp.sin(t * theta) / sin_theta)
    
    
    # Return the interpolated point
    return a * P0 + b * P1

def velocity(P0, P1, t):
    # Normalize P0 and P1 to ensure they are on the sphere S2
    P0 = jnp.nan_to_num(P0 / jnp.linalg.norm(P0), nan = 1/jnp.sqrt(P0.shape[-1]))
    P1 = jnp.nan_to_num(P1 / jnp.linalg.norm(P1), nan = 1/jnp.sqrt(P1.shape[-1]))
    # Compute the cosine of the angle between P0 and P1
    cos_theta = jnp.dot(P0, P1)
    
    # Clip cos_theta to avoid numerical issues with acos
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle theta
    theta = jnp.arccos(cos_theta)
    
    # Compute the sin of theta
    sin_theta = jnp.sin(theta)

    a = jnp.where(sin_theta < 1e-6, -1, -theta * jnp.cos((1 - t) * theta) / sin_theta)
    b = jnp.where(sin_theta < 1e-6, 1, theta * jnp.cos(t * theta) / sin_theta)


    # SLERP velocity formula
    # Return the tangent velocity
    return a * P0 + b * P1

def tangent_norm(v, w, p):
    """
    Compute the distance between two tangent vectors v and w at point p on the sphere.
    First ensures both vectors are truly tangent by projecting out any radial component.
    """

    p = jnp.nan_to_num(p / jnp.linalg.norm(p), nan = 1/jnp.sqrt(p.shape[-1]))
    # Project both vectors onto tangent space at p (if they're not already tangent)
    v_tangent = v - jnp.dot(v, p) * p
    w_tangent = w - jnp.dot(w, p) * p
    
    # Compute the Euclidean distance between the tangent vectors
    return jnp.mean(jnp.square(v_tangent - w_tangent))

def exponential_map(p, v, delta_t):
    """
    Perform a step on the sphere S2 using the exponential map.
    
    Args:
    p (array): Current point on the sphere (3D unit vector)
    v (array): Velocity vector (tangent to the sphere at p)
    delta_t (float): Time step
    
    Returns:
    array: New point on the sphere after the step
    """
    p = jnp.asarray(p)
    v = jnp.asarray(v)
    
    # Ensure p is normalized
    p = p / jnp.linalg.norm(p)
    
    # Project v onto the tangent space of p (should already be tangent, but this ensures numerical stability)
    v = v - jnp.dot(v, p) * p
    
    # Compute the norm of v
    v_norm = jnp.linalg.norm(v)
    
    # Handle the case where v is very small
    def small_step():
        # For very small v, we can approximate exp(v) ≈ p + v
        step = p + delta_t * v
        return step / jnp.linalg.norm(step)  # Normalize to ensure we stay on the sphere
    
    # Handle the general case
    def general_step():
        theta = v_norm * delta_t
        return jnp.cos(theta) * p + jnp.sin(theta) * (v / v_norm)
    
    # Choose between small step and general step based on the magnitude of v
    new_p = jnp.where(v_norm < 1e-6, small_step(), general_step())
    
    return new_p

