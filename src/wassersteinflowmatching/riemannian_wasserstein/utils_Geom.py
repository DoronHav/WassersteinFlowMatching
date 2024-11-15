import jax.numpy as jnp # type: ignore


class torus:

    def project_to_geometry(self, P):
        # For n-dimensional torus, points are represented as n angles
        # Project by taking modulo 2π for all angles
        return jnp.mod(P, 2 * jnp.pi)

    def distance(self, P0, P1):
        # Normalize angles to [0, 2π)
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Calculate shortest angular distances for all dimensions
        diff = jnp.minimum(
            jnp.abs(P0 - P1),
            2 * jnp.pi - jnp.abs(P0 - P1)
        )
        
        # Return geodesic distance on n-torus
        return jnp.sqrt(jnp.sum(diff**2))

    def distance_matrix(self, P0, P1):
        # Normalize angles to [0, 2π)
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Calculate shortest angular distances for all dimensions
        diff = jnp.minimum(
            jnp.abs(P0[:, None, :] - P1[None, :, :]),
            2 * jnp.pi - jnp.abs(P0[:, None, :] - P1[None, :, :])
        )
        
        # Return matrix of distances
        return jnp.sqrt(jnp.sum(diff**2, axis=-1))

    def velocity(self, P0, P1, t):
        """
        Velocity vector at time t along the geodesic from P0 to P1.
        For torus, this is constant and equal to the logarithmic map.
        """
        return jnp.arctan2(jnp.sin(P1 - P0), jnp.cos(P1 - P0))

    def tangent_norm(self, v, w, p):
        """
        Compute the distance between two tangent vectors v and w at point p on the n-torus.
        On a torus, tangent vectors are just n-dimensional vectors as we're working in angle space.
        """
        # For torus in angle coordinates, all vectors are already tangent
        # Compute Euclidean distance in angle space
        return jnp.mean(jnp.square(v - w))

    def exponential_map(self, p, v, delta_t):
        """
        Perform a step on the n-torus using the exponential map.
        In angle coordinates, this is just linear motion with wraparound.
        
        Args:
        p (array): Current point on n-torus (n-dimensional array of angles)
        v (array): Velocity vector (n-dimensional array of angular velocities)
        delta_t (float): Time step
        
        Returns:
        array: New point on n-torus after the step
        """
        # Simple linear motion in angle space with wraparound
        new_p = p + v * delta_t
        return self.project_to_geometry(new_p)

    def interpolant(self, P0, P1, t):
        """
        Geodesic interpolation between two points on the d-dimensional torus.
        """
        log = self.velocity(P0, P1, 0)
        return self.exponential_map(P0, log, t)

class sphere:
    def project_to_geometry(self, P):
        # Normalize bath of points P to ensure it is on the sphere Sd
        return jnp.nan_to_num(P /  jnp.linalg.norm(P, axis=-1, keepdims=True), nan = 1/jnp.sqrt(P.shape[-1]))


    def distance(self, P0, P1):
        # Normalize P0 and P1 to ensure they are on the sphere S2
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Compute the dot product between the two points
        dot_product = jnp.dot(P0, P1)
        
        # Clip the dot product to avoid numerical issues with arccos
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        
        # Compute the great circle distance (angular distance)
        return jnp.arccos(dot_product)

    def distance_matrix(self, P0, P1):
        # Normalize the points to ensure they are on the sphere S2
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Compute the dot product matrix
        dot_product_matrix = P0 @ P1.T
        
        # Clip the dot product matrix to avoid numerical issues with arccos
        dot_product_matrix = jnp.clip(dot_product_matrix, -1.0, 1.0)
        
        # Compute the great circle distance matrix
        return jnp.arccos(dot_product_matrix)

    def interpolant(self, P0, P1, t):
        # Normalize P0 and P1 to ensure they are on the sphere S2
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
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

    def velocity(self, P0, P1, t):
        # Normalize P0 and P1 to ensure they are on the sphere S2
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
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

    def tangent_norm(self, v, w, p):
        """
        Compute the distance between two tangent vectors v and w at point p on the sphere.
        First ensures both vectors are truly tangent by projecting out any radial component.
        """

        p = self.project_to_geometry(p)
        # Project both vectors onto tangent space at p (if they're not already tangent)
        v_tangent = v - jnp.dot(v, p) * p
        w_tangent = w - jnp.dot(w, p) * p
        
        # Compute the Euclidean distance between the tangent vectors
        return jnp.mean(jnp.square(v_tangent - w_tangent))

    def exponential_map(self, p, v, delta_t):
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
        p = self.project_to_geometry(p)
        
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

class hyperbolic:
    def project_to_geometry(self, P):
        # Project points to ensure they lie within the Poincaré ball
        # Normalize points that lie outside the unit ball
        norm = jnp.linalg.norm(P, axis=-1, keepdims=True)
        return jnp.where(norm >= 1.0, P / (norm + 1e-5), P)

    def mobius_addition(self, x, y):
        """
        Möbius addition in the Poincaré ball model.
        Formula: (1 + 2<x,y> + |y|²)x + (1 - |x|²)y / (1 + 2<x,y> + |x|²|y|²)
        """
        x_norm_sq = jnp.sum(x**2)
        y_norm_sq = jnp.sum(y**2)
        dot_prod = jnp.dot(x, y)
        numerator = (1 + 2*dot_prod + y_norm_sq)*x + (1 - x_norm_sq)*y
        denominator = 1 + 2*dot_prod + x_norm_sq*y_norm_sq
        return numerator / denominator

    def mobius_addition_batch(self, x, y):
        """
        Vectorized Möbius addition for batches of points.
        x: shape (n, d) or (d,)
        y: shape (m, d) or (d,)
        Returns: shape (n, m, d) or (n, d) depending on input shapes
        """
        # Add batch dimensions if needed
        if x.ndim == 1:
            x = x[None, :]
        if y.ndim == 1:
            y = y[None, :]
        
        # Reshape for broadcasting
        x = x[:, None, :]  # (n, 1, d)
        y = y[None, :, :]  # (1, m, d)
        
        # Compute norms and dot products
        x_norm_sq = jnp.sum(x**2, axis=-1, keepdims=True)  # (n, 1, 1)
        y_norm_sq = jnp.sum(y**2, axis=-1, keepdims=True)  # (1, m, 1)
        dot_prod = jnp.sum(x * y, axis=-1, keepdims=True)  # (n, m, 1)
        
        # Compute Möbius addition
        numerator = (1 + 2*dot_prod + y_norm_sq)*x + (1 - x_norm_sq)*y
        denominator = 1 + 2*dot_prod + x_norm_sq*y_norm_sq
        
        return numerator / denominator

    def distance(self, P0, P1):
        """
        Compute the hyperbolic distance between two points in the Poincaré ball.
        Formula: d(x,y) = 2 * arctanh(|(-x) ⊕ y|)
        """
        # Project points to ensure they're in the unit ball
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Compute the Möbius addition of -P0 and P1
        minus_p0 = -P0
        mobius_sum = self.mobius_addition(minus_p0, P1)
        
        # Compute the norm of the result
        norm = jnp.linalg.norm(mobius_sum)
        
        # Clip to avoid numerical issues
        norm = jnp.clip(norm, 0.0, 1.0 - 1e-5)
        
        # Return the hyperbolic distance
        return 2 * jnp.arctanh(norm)

    def distance_matrix(self, P0, P1):
        """
        Compute pairwise hyperbolic distances between two sets of points.
        P0: shape (n, d)
        P1: shape (m, d)
        Returns: shape (n, m)
        """
        # Project points to ensure they're in the unit ball
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Compute the Möbius addition of -P0 and P1 for all pairs
        minus_P0 = -P0
        mobius_sums = self.mobius_addition_batch(minus_P0, P1)  # shape (n, m, d)
        
        # Compute the norms of the results
        norms = jnp.linalg.norm(mobius_sums, axis=-1)  # shape (n, m)
        
        # Clip to avoid numerical issues
        norms = jnp.clip(norms, 0.0, 1.0 - 1e-5)
        
        # Return the hyperbolic distances
        return 2 * jnp.arctanh(norms)

    def interpolant(self, P0, P1, t):
        """
        Compute geodesic interpolation in the Poincaré ball.
        This is the geodesic from P0 to P1 at time t.
        """
        # Project points to ensure they're in the unit ball
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # If points are very close, return linear interpolation
        if jnp.allclose(P0, P1):
            return (1 - t) * P0 + t * P1
        
        
        # Compute the geodesic
        P0_norm = jnp.linalg.norm(P0)
        P1_norm = jnp.linalg.norm(P1)
        
        # Handle special cases
        if P0_norm < 1e-6:  # P0 is near origin
            return t * P1
        if P1_norm < 1e-6:  # P1 is near origin
            return (1 - t) * P0
            
        # General case: compute the geodesic using the exponential map
        initial_velocity = self.log_map(P0, P1)
        return self.exponential_map(P0, initial_velocity, t)

    def velocity(self, P0, P1, t):
        """
        Compute the velocity vector at time t along the geodesic from P0 to P1.
        
        Args:
            P0: Starting point in the Poincaré ball
            P1: Ending point in the Poincaré ball
            t: Time parameter in [0,1]
            
        Returns:
            Velocity vector at the point gamma(t) where gamma is the geodesic from P0 to P1
        """
        # Project points to ensure they're in the unit ball
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # If points are very close, return zero velocity
        if jnp.allclose(P0, P1):
            return jnp.zeros_like(P0)
        
        # Compute the initial velocity using the log map
        initial_velocity = self.log_map(P0, P1)
        
        # Get the point at time t along the geodesic
        Pt = self.interpolant(P0, P1, t)
        
        # Compute the conformal factors
        lambda_P0 = 2 / (1 - jnp.sum(P0**2))
        lambda_Pt = 2 / (1 - jnp.sum(Pt**2))
        
        # Compute the parallel transport from P0 to Pt
        # First, get the squared norms
        P0_norm_sq = jnp.sum(P0**2)
        Pt_norm_sq = jnp.sum(Pt**2)
        
        # Compute the inner product
        inner_prod = jnp.sum(P0 * Pt)
        
        # Compute the parallel transport scaling factor
        # This accounts for the change in the metric tensor along the geodesic
        scaling = lambda_P0 / lambda_Pt * (
            (1 - P0_norm_sq) / (1 - Pt_norm_sq) * 
            (1 + 2 * inner_prod + Pt_norm_sq) / 
            (1 + 2 * inner_prod + P0_norm_sq)
        )
        
        # For numerical stability, clip the scaling factor
        scaling = jnp.clip(scaling, 1e-6, 1e6)
        
        # Parallel transport the initial velocity to Pt
        transported_velocity = scaling * initial_velocity
        
        # Project the transported velocity onto the tangent space at Pt
        # This ensures the velocity remains tangent to the manifold
        Pt_component = jnp.sum(transported_velocity * Pt) * Pt
        tangent_velocity = transported_velocity - Pt_component
        
        return tangent_velocity

    def tangent_norm(self, v, w, p):
        """
        Compute the norm of the difference between two tangent vectors v and w at point x
        in the Poincaré ball model.
        
        Args:
            v: First tangent vector
            w: Second tangent vector
            x: Base point in the Poincaré ball where these vectors are tangent
            
        Returns:
            The squared norm of the difference between the tangent vectors
            under the hyperbolic metric
        """
        # Project base point to ensure it's in the unit ball
        p = self.project_to_geometry(p)
        
        # Ensure vectors are tangent by projecting out radial components
        p_norm_sq = jnp.sum(p**2)
        
        # Project v onto tangent space at x
        v_dot_p = jnp.sum(v * p)
        v_tangent = v - (v_dot_p * p)
        
        # Project w onto tangent space at x
        w_dot_p = jnp.sum(w * p)
        w_tangent = w - (w_dot_p * p)
        
        # Compute the difference between the tangent vectors
        diff = v_tangent - w_tangent
        
        # Compute the conformal factor (hyperbolic metric tensor)
        # In the Poincaré ball, the metric tensor is scaled by lambda_x^2
        lambda_p = 2 / (1 - p_norm_sq)
        
        # Compute the squared norm under the hyperbolic metric
        # ||v||^2 = <v,v>_x = λ_x^2 <v,v>_euclidean
        return lambda_p**2 * jnp.sum(diff**2)
    
    def exponential_map(self, p, v, delta_t=1.0):
        """
        Compute the exponential map in the Poincaré ball.
        Maps a tangent vector v at point p to a point in the manifold.
        """
        # Project p to ensure it's in the unit ball
        p = self.project_to_geometry(p)
        
        # Compute the norm of v
        v_norm = jnp.linalg.norm(v)
        
        # If v is very small, return p
        if v_norm < 1e-6:
            return p
        
        # Compute the conformal factor
        lambda_p = 2 / (1 - jnp.sum(p**2))
        
        # Scale the vector by delta_t
        v = v * delta_t
        
        # Compute the exponential map
        v_norm = jnp.linalg.norm(v)
        coef = jnp.tanh(v_norm / (2 * lambda_p)) / v_norm
        
        # Return the result
        result = self.mobius_addition(p, coef * v)
        return self.project_to_geometry(result)

    def log_map(self, p, q):
        """
        Compute the logarithmic map in the Poincaré ball.
        Maps a point q to a tangent vector at point p.
        """
        # Project points to ensure they're in the unit ball
        p = self.project_to_geometry(p)
        q = self.project_to_geometry(q)
        
        # If points are very close, return zero vector
        if jnp.allclose(p, q):
            return jnp.zeros_like(p)
        
        # Compute the Möbius addition of -p and q
        minus_p = -p
        mobius_diff = self.mobius_addition(minus_p, q)
        
        # Compute the norm of the difference
        diff_norm = jnp.linalg.norm(mobius_diff)
        
        # Compute the conformal factor
        lambda_p = 2 / (1 - jnp.sum(p**2))
        
        # Compute the logarithmic map
        return 2 * lambda_p * jnp.arctanh(diff_norm) * mobius_diff / diff_norm
