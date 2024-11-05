import jax.numpy as jnp # type: ignore

def distance(P0, P1):
    """
    Compute the hyperbolic distance between two points in the Poincaré disk model.
    Points should be in the unit disk (||P|| < 1).
    """
    # Ensure points are within the unit disk
    P0_norm = jnp.linalg.norm(P0)
    P1_norm = jnp.linalg.norm(P1)
    
    # Project points that might be at or beyond the boundary slightly inside
    P0 = jnp.where(P0_norm >= 1.0, P0 * 0.99 / P0_norm, P0)
    P1 = jnp.where(P1_norm >= 1.0, P1 * 0.99 / P1_norm, P1)
    
    # Compute the hyperbolic distance using the Poincaré disk formula
    num = 2 * jnp.linalg.norm(P0 - P1)**2
    den = (1 - jnp.linalg.norm(P0)**2) * (1 - jnp.linalg.norm(P1)**2)
    
    # Avoid numerical issues with arcosh
    argument = 1 + num/den
    argument = jnp.clip(argument, 1 + 1e-7, float('inf'))
    
    return jnp.arccosh(argument)

def matrix(P0, P1):
    """
    Compute the matrix of hyperbolic distances between two sets of points.
    """
    # Ensure points are within the unit disk
    P0_norm = jnp.linalg.norm(P0, axis=1, keepdims=True)
    P1_norm = jnp.linalg.norm(P1, axis=1, keepdims=True)
    
    P0 = jnp.where(P0_norm >= 1.0, P0 * 0.99 / P0_norm, P0)
    P1 = jnp.where(P1_norm >= 1.0, P1 * 0.99 / P1_norm, P1)
    
    # Compute pairwise differences
    diff_matrix = P0[:, None, :] - P1[None, :, :]
    norm_diff_matrix = jnp.linalg.norm(diff_matrix, axis=2)**2
    
    # Compute denominator terms
    P0_term = 1 - jnp.sum(P0**2, axis=1)[:, None]
    P1_term = 1 - jnp.sum(P1**2, axis=1)[None, :]
    
    # Compute distance matrix
    argument = 1 + (2 * norm_diff_matrix) / (P0_term * P1_term)
    argument = jnp.clip(argument, 1 + 1e-7, float('inf'))
    
    return jnp.arccosh(argument)

def interpolant(P0, P1, t):
    """
    Compute the geodesic interpolation between two points in hyperbolic space.
    """
    # Ensure points are within the unit disk
    P0_norm = jnp.linalg.norm(P0)
    P1_norm = jnp.linalg.norm(P1)
    
    P0 = jnp.where(P0_norm >= 1.0, P0 * 0.99 / P0_norm, P0)
    P1 = jnp.where(P1_norm >= 1.0, P1 * 0.99 / P1_norm, P1)
    
    # Compute the hyperbolic distance
    dist = distance(P0, P1)
    
    # Handle the case where points are very close
    def small_dist():
        step = (1-t)*P0 + t*P1
        step_norm = jnp.linalg.norm(step)
        return jnp.where(step_norm >= 1.0, step * 0.99 / step_norm, step)
    
    # Handle the general case using geodesic interpolation
    def general_dist():
        P0_norm = jnp.linalg.norm(P0)
        P1_norm = jnp.linalg.norm(P1)
        
        # Compute normalized vectors and interpolate in the hyperbolic space
        result = jnp.tanh((1-t)*jnp.arctanh(P0_norm) + t*jnp.arctanh(P1_norm)) * \
                ((1-t)*P0/P0_norm + t*P1/P1_norm)
        
        # Ensure the result stays within the unit disk
        result_norm = jnp.linalg.norm(result)
        return jnp.where(result_norm >= 1.0, result * 0.99 / result_norm, result)
    
    return jnp.where(dist < 1e-6, small_dist(), general_dist())

def velocity(P0, P1, t):
    """
    Compute the velocity vector of the geodesic at time t in hyperbolic space.
    """
    # Ensure points are within the unit disk
    P0_norm = jnp.linalg.norm(P0)
    P1_norm = jnp.linalg.norm(P1)
    
    P0 = jnp.where(P0_norm >= 1.0, P0 * 0.99 / P0_norm, P0)
    P1 = jnp.where(P1_norm >= 1.0, P1 * 0.99 / P1_norm, P1)
    
    # Compute the point at time t
    Pt = interpolant(P0, P1, t)
    
    # Compute the conformal factor for the hyperbolic metric
    conformal_factor = 2 / (1 - jnp.sum(Pt**2))
    
    # Compute the Euclidean velocity and scale it by the conformal factor
    euclidean_velocity = P1 - P0
    return conformal_factor * euclidean_velocity

def tangent_norm(v, w, p):
    """
    Compute the hyperbolic norm between two tangent vectors v and w at point p.
    """
    # Ensure p is within the unit disk
    p_norm = jnp.linalg.norm(p)
    p = jnp.where(p_norm >= 1.0, p * 0.99 / p_norm, p)
    
    # Compute the conformal factor for the hyperbolic metric
    conformal_factor = 2 / (1 - jnp.sum(p**2))
    
    # Project vectors onto the tangent space if needed
    v_tangent = v - jnp.dot(v, p) * p
    w_tangent = w - jnp.dot(w, p) * p
    
    # Compute the hyperbolic inner product
    return conformal_factor**2 * jnp.mean(jnp.square(v_tangent - w_tangent))

def exponential_map(p, v, delta_t):
    """
    Perform a step in hyperbolic space using the exponential map.
    """
    p = jnp.asarray(p)
    v = jnp.asarray(v)
    
    # Ensure p is within the unit disk
    p_norm = jnp.linalg.norm(p)
    p = jnp.where(p_norm >= 1.0, p * 0.99 / p_norm, p)
    
    # Project v onto the tangent space
    v = v - jnp.dot(v, p) * p
    
    # Compute the norm of v in hyperbolic metric
    v_norm = jnp.linalg.norm(v) * 2 / (1 - jnp.sum(p**2))
    
    def small_step():
        # For small steps, use linear approximation
        step = p + delta_t * v
        step_norm = jnp.linalg.norm(step)
        return jnp.where(step_norm >= 1.0, step * 0.99 / step_norm, step)
    
    def general_step():
        # For general steps, use the hyperbolic exponential map
        scaled_v = v * delta_t
        v_norm = jnp.linalg.norm(scaled_v)
        result = jnp.cosh(v_norm) * p + jnp.sinh(v_norm) * scaled_v / v_norm
        
        # Ensure the result stays within the unit disk
        result_norm = jnp.linalg.norm(result)
        return jnp.where(result_norm >= 1.0, result * 0.99 / result_norm, result)
    
    return jnp.where(v_norm * delta_t < 1e-6, small_step(), general_step())