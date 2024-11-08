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

