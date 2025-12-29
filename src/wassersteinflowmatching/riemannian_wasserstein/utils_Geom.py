import jax.numpy as jnp # type: ignore
import numpy as np

class euclidean:
    def project_to_geometry(self, P, use_cpu=False):
        return P

    def distance(self, P0, P1):
        return jnp.sum((P0 - P1)**2)

    def distance_matrix(self, P0, P1):
        return jnp.sum((P0[:, None, :] - P1[None, :, :])**2, axis=-1)

    def velocity(self, P0, P1, t):
        return P1 - P0

    def tangent_norm(self, v, w, p):
        return jnp.mean(jnp.square(v - w))

    def exponential_map(self, p, v, delta_t):
        return p + v * delta_t

    def interpolant(self, P0, P1, t):
        return (1 - t) * P0 + t * P1
    
    def weighted_mean(self, points, weights):
        weights = weights / (jnp.sum(weights) + 1e-9)
        return jnp.sum(points * weights[:, None], axis=0)


class torus:

    def project_to_geometry(self, P, use_cpu=False):
        # For n-dimensional torus, points are represented as n angles
        # Project by taking modulo 2π for all angles
        if use_cpu:
            return np.mod(P, 2 * np.pi)
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
        return jnp.sum(diff**2)

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
        return jnp.sum(diff**2, axis=-1)

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
    
    def weighted_mean(self, points, weights):
        """
        Computes the weighted Fréchet mean on the torus using circular statistics.
        
        It maps angles to unit vectors, averages the vectors, and maps back to angles.
        This respects the periodicity (e.g., mean of 0.1 and 2π-0.1 is 0, not π).
        
        Args:
            points: Array of shape (n, d) containing points on the torus [0, 2π]
            weights: Array of shape (n,) containing weights for each point
            
        Returns:
            Weighted mean point projected onto the torus in range [0, 2π]
        """
        # 1. Normalize weights to sum to 1
        # Shape: (n,) -> (n, 1) for broadcasting against (n, d) points
        weights = weights / (jnp.sum(weights) + 1e-9)
        weights_expanded = weights[:, None] 
        
        # 2. Convert angles to unit vectors (phasors)
        # We process each dimension d independently as S^1
        sin_components = jnp.sin(points)
        cos_components = jnp.cos(points)
        
        # 3. Compute weighted center of mass for the vectors
        # Result shape: (d,)
        mean_sin = jnp.sum(sin_components * weights_expanded, axis=0)
        mean_cos = jnp.sum(cos_components * weights_expanded, axis=0)
        
        # 4. Recover the mean angle using arctan2
        # arctan2 returns values in [-π, π]
        mean_angles = jnp.arctan2(mean_sin, mean_cos)
        
        # 5. Project back to [0, 2π] geometry
        return self.project_to_geometry(mean_angles)
    
class sphere:
    def project_to_geometry(self, P, use_cpu=False):
        # Normalize bath of points P to ensure it is on the sphere Sd
        if use_cpu:
            return np.nan_to_num(P /  np.linalg.norm(P, axis=-1, keepdims=True), nan = 1/np.sqrt(P.shape[-1]))
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
        return jnp.arccos(dot_product)**2

    def distance_matrix(self, P0, P1):
        # Normalize the points to ensure they are on the sphere S2
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Compute the dot product matrix
        dot_product_matrix = P0 @ P1.T
        
        # Clip the dot product matrix to avoid numerical issues with arccos
        dot_product_matrix = jnp.clip(dot_product_matrix, -1.0, 1.0)
        
        # Compute the great circle distance matrix
        return jnp.arccos(dot_product_matrix)**2

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
    
    def weighted_mean(self, points, weights):
        """
        Compute weighted mean on the sphere using Euclidean mean followed by projection.
        
        Args:
            points: Array of shape (n, d) containing points on the sphere
            weights: Array of shape (n,) containing weights for each point
            
        Returns:
            Weighted mean point projected onto the sphere
        """
        # Normalize weights
        weights = weights / (jnp.sum(weights) + 1e-9)
        
        # Compute weighted Euclidean mean
        euclidean_mean = jnp.sum(points * weights[:, None], axis=0)
        
        # Project back onto sphere (normalize to unit length)
        return self.project_to_geometry(euclidean_mean)

class hyperbolic:
    """
    Implements Hyperbolic geometry using the Lorentz (Hyperboloid) model.
    Points are represented in R^(d+1) such that <x, x>_L = -1 and x[0] > 0.
    """
    

    def _minkowski_dot(self, x, y):
        """Internal helper for Minkowski inner product: -x0*y0 + x1*y1 + ..."""
        # We assume the 0-th index is the 'time' component with negative signature
        res = -x[..., 0] * y[..., 0] + jnp.sum(x[..., 1:] * y[..., 1:], axis=-1)
        return res

    def project_to_geometry(self, P, use_cpu=False):
        """
        Project points onto the upper sheet of the hyperboloid.
        Method: Normalize by Minkowski norm, then flip sign if on lower sheet.
        """
        if use_cpu:
  
            # Calculate Minkowski squared norm: <P, P>_L
            # Inline _minkowski_dot for numpy
            x_space = P[..., 1:]
            
            # 2. Calculate the squared norm of the spatial part
            spatial_sq_norm = np.sum(x_space**2, axis=-1, keepdims=True)
            
            # 3. Solve for x0 such that -x0^2 + spatial_sq_norm = -1
            # implies x0^2 = 1 + spatial_sq_norm
            x0 = np.sqrt(1.0 + spatial_sq_norm)
            
            # 4. Reassemble the vector [x0, x_space]
            P_proj = np.concatenate([x0, x_space], axis=-1)
            
            return P_proj

    # 1. Extract the spatial components (everything after index 0)
        x_space = P[..., 1:]
        
        # 2. Calculate the squared norm of the spatial part
        spatial_sq_norm = jnp.sum(x_space**2, axis=-1, keepdims=True)
        
        # 3. Solve for x0 such that -x0^2 + spatial_sq_norm = -1
        # implies x0^2 = 1 + spatial_sq_norm
        x0 = jnp.sqrt(1.0 + spatial_sq_norm)
        
        # 4. Reassemble the vector [x0, x_space]
        P_proj = jnp.concatenate([x0, x_space], axis=-1)
        
        return P_proj

    def distance(self, P0, P1):
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Hyperbolic distance formula: arccosh(-<x, y>_L)
        inner_prod = self._minkowski_dot(P0, P1)
        
        # Clamp for numerical stability: inner product must be <= -1
        inner_prod = jnp.minimum(inner_prod, -1.0 - 1e-7)
        
        return jnp.arccosh(-inner_prod)**2

    def distance_matrix(self, P0, P1):
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Vectorized Minkowski Product
        # <P0, P1>_L = -P0_0*P1_0^T + P0_space @ P1_space^T
        term_time = -jnp.outer(P0[:, 0], P1[:, 0])
        term_space = P0[:, 1:] @ P1[:, 1:].T
        inner_prod_mat = term_time + term_space
        
        inner_prod_mat = jnp.minimum(inner_prod_mat, -1.0 - 1e-7)
        return jnp.arccosh(-inner_prod_mat)**2

    def interpolant(self, P0, P1, t):
        """
        Hyperbolic Linear Interpolation (analogue to SLERP on sphere).
        """
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Cosine rule analogue: <P0, P1>_L = -cosh(dist)
        inner_prod = self._minkowski_dot(P0, P1)
        inner_prod = jnp.minimum(inner_prod, -1.0 - 1e-7)
        
        # The distance (angle) between points
        omega = jnp.arccosh(-inner_prod)
        
        sinh_omega = jnp.sinh(omega)
        
        # Handle case where points are effectively identical (omega ~ 0)
        # Standard SLERP formula adapted with sinh instead of sin
        a = jnp.where(sinh_omega < 1e-6, 1.0 - t, jnp.sinh((1 - t) * omega) / sinh_omega)
        b = jnp.where(sinh_omega < 1e-6, t, jnp.sinh(t * omega) / sinh_omega)
        
        # Reshape for broadcasting
        a = a[..., None]
        b = b[..., None]
        
        return a * P0 + b * P1

    def velocity(self, P0, P1, t):
        """
        Analytical derivative of the interpolant with respect to t.
        """
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        inner_prod = self._minkowski_dot(P0, P1)
        inner_prod = jnp.minimum(inner_prod, -1.0 - 1e-7)
        
        omega = jnp.arccosh(-inner_prod)
        sinh_omega = jnp.sinh(omega)
        
        # Derivative of the sinh coefficients
        # d/dt [sinh((1-t)w)/sinh(w)] = -w * cosh((1-t)w)/sinh(w)
        # d/dt [sinh(tw)/sinh(w)]     =  w * cosh(tw)/sinh(w)
        
        # term a derivative
        da = jnp.where(sinh_omega < 1e-6, -1.0, -omega * jnp.cosh((1 - t) * omega) / sinh_omega)
        # term b derivative
        db = jnp.where(sinh_omega < 1e-6, 1.0, omega * jnp.cosh(t * omega) / sinh_omega)
        
        da = da[..., None]
        db = db[..., None]
        
        return da * P0 + db * P1

    def tangent_norm(self, v, w, p):
        """
        Norm in the tangent space at point p.
        The tangent space T_p H is the set of vectors orthogonal to p under Minkowski metric.
        However, the restriction of the Minkowski metric to T_p H is POSITIVE DEFINITE.
        So this looks like a standard Euclidean squared norm, but computed using Minkowski dot.
        """
        # First, ensure v and w are tangent to p
        # Project vector u onto tangent space: u_tan = u + <u, p>_L * p
        # (Note the plus sign because <p,p>_L = -1)
        
        p = self.project_to_geometry(p)
        
        v_dot_p = self._minkowski_dot(v, p)[..., None]
        w_dot_p = self._minkowski_dot(w, p)[..., None]
        
        v_tan = v + v_dot_p * p
        w_tan = w + w_dot_p * p
        
        diff = v_tan - w_tan
        
        # The norm squared of a tangent vector is <diff, diff>_L
        # Since it's tangent, this value will be positive.
        return jnp.mean(self._minkowski_dot(diff, diff))

    def exponential_map(self, p, v, delta_t):
        """
        Hyperbolic Exponential Map.
        Moves point p in direction v by time delta_t.
        """
        p = self.project_to_geometry(p)
        
        # Project v to tangent space to be safe
        v_dot_p = self._minkowski_dot(v, p)[..., None]
        v = v + v_dot_p * p
        
        # Calculate Minkowski norm of velocity vector
        # For tangent vectors, <v, v>_L >= 0
        v_sq_norm = self._minkowski_dot(v, v)
        v_norm = jnp.sqrt(jnp.maximum(v_sq_norm, 0.0))
        
        # Formula: p * cosh(norm * t) + (v/norm) * sinh(norm * t)
        
        def small_step():
             # Approximation for small velocity
             step = p + v * delta_t
             return self.project_to_geometry(step)

        def general_step():
            norm_val = v_norm * delta_t
            # Reshape for broadcasting
            cosh_term = jnp.cosh(norm_val)[..., None]
            sinh_term = jnp.sinh(norm_val)[..., None]
            
            # v / v_norm term (handle div by zero safely)
            direction = v / (v_norm[..., None] + 1e-9)
            
            return p * cosh_term + direction * sinh_term

        # Select based on velocity magnitude
        res = jnp.where(v_norm[..., None] < 1e-6, small_step(), general_step())
        return res

    def weighted_mean(self, points, weights):
        """
        Lorentz Centroid.
        1. Compute weighted sum in embedding space R^(d+1).
        2. Project result back to Hyperboloid.
        """
        weights = weights / (jnp.sum(weights) + 1e-9)
        
        # Linear sum in embedding space
        linear_mean = jnp.sum(points * weights[:, None], axis=0)
        
        # Project back to manifold
        return self.project_to_geometry(linear_mean)
    
class so3:
    # We use quaternions representation of SO(3) for simplicity
    # A point cloud has a shape (p,n,4) where each point is a unit quaternion
    def project_to_geometry(self, P, use_cpu=False):
        # Project points to ensure they are normalized
        return P /  jnp.linalg.norm(P, axis=-1, keepdims=True) 

    def quaternion_conjugate(self, q):
        # q = [w, x, y, z]
        return jnp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]],axis=-1)

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return jnp.stack([w, x, y, z], axis=-1)

    def velocity(self, P0, P1, t):
        """
        velocity is the rotation vector (in R3)
        """
        # Normalize P0 and P1 to ensure they are on SO(3)
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)

        # Ensure shortest path (handle double cover)
        P1 = jnp.where(jnp.sum(P0 * P1, axis=-1, keepdims=True) < 0, P1 * -1, P1)
        
        # Compute the relative rotation quaternion
        q_rel = self.quaternion_multiply(self.quaternion_conjugate(P0), P1)
        
        # Convert to axis-angle representation
        angle = jnp.arccos(jnp.clip(q_rel[..., 0], -1.0, 1.0))
        axis = q_rel[..., 1:] / jnp.linalg.norm(q_rel[..., 1:], axis=-1, keepdims=True)

        epsilon_vec = angle[...,None] * axis
        
        return epsilon_vec
    
    def exponential_map(self, p, v, delta_t):
        # p: quaternion [w, x, y, z]
        # v: angular velocity vector [omega, vx, vy, vz]
        # delta_t: time step
        theta = jnp.linalg.norm(v, axis=-1)[...,None]
        theta = jnp.where(theta < 1e-8, 0, theta)  # avoid division by zero

        u = v / theta
        alpha = delta_t * theta

        step = jnp.concatenate([jnp.cos(alpha), jnp.sin(alpha) * u],axis=-1)

        q_rot = self.quaternion_multiply(p, step)
        return self.project_to_geometry(q_rot)

    def interpolant(self, P0, P1, t):
        # Normalize P0 and P1 to ensure they are on SO(3)
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        velocity = self.velocity(P0, P1, t=0)
        q_interp = self.exponential_map(P0, velocity, t)
        return q_interp
    
    def tangent_norm(self, v, w):
        # Simply use Euclidean norm in axis-angle representation
        return jnp.mean(jnp.square(v - w))

    def distance(self, P0, P1):
        # Normalize P0 and P1
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)

        # Ensure shortest path
        P1 = jnp.where(jnp.sum(P0 * P1, axis=-1, keepdims=True) < 0, P1 * -1, P1)

        # Relative rotation quaternion
        q_rel = self.quaternion_multiply(self.quaternion_conjugate(P0), P1)

        # Angle of rotation
        angle = jnp.arccos(jnp.clip(q_rel[..., 0], -1.0, 1.0))

        return angle**2

    def distance_matrix(self, P0, P1):
        # Normalize P0 and P1
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)

        # Compute all pairwise quaternion products
        # This is tricky with vectorized quaternion multiply
        # Let's compute the angle directly from dot product
        dots = jnp.sum(P0[:, None, :] * P1[None, :, :], axis=-1)
        # Adjust for shortest path
        dots_adj = jnp.where(dots < 0, -dots, dots)
        # Angle is arccos of absolute dot product (since unit quaternions)
        angles = jnp.arccos(jnp.clip(dots_adj, -1.0, 1.0))
        return angles**2

class se3:
    # SE(3) representation: [qw, qx, qy, qz, tx, ty, tz]
    # A point cloud has shape (p, n, 7) where each point is a quaternion + translation
    def project_to_geometry(self, P, use_cpu=False):
        # Normalize quaternion part (first 4 components)
        q_norm = jnp.linalg.norm(P[..., :4], axis=-1, keepdims=True)
        q_normalized = P[..., :4] / q_norm
        # Translation part remains unchanged
        return jnp.concatenate([q_normalized, P[..., 4:]], axis=-1)

    def quaternion_conjugate(self, q):
        # q = [w, x, y, z]
        return jnp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return jnp.stack([w, x, y, z], axis=-1)

    def rotate_vector_by_quaternion(self, q, v):
        # Rotate vector v by quaternion q
        q_conj = self.quaternion_conjugate(q)
        v_quat = jnp.concatenate([jnp.zeros_like(v[..., :1]), v], axis=-1)
        rotated = self.quaternion_multiply(self.quaternion_multiply(q, v_quat), q_conj)
        return rotated[..., 1:]

    def velocity(self, P0, P1, t):
        """
        Velocity is the twist: [angular velocity (3), linear velocity (3)]
        """
        # Normalize P0 and P1
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)

        # Extract rotations and translations
        q0, t0 = P0[..., :4], P0[..., 4:]
        q1, t1 = P1[..., :4], P1[..., 4:]

        # Ensure shortest path for rotation (handle double cover)
        q1 = jnp.where(jnp.sum(q0 * q1, axis=-1, keepdims=True) < 0, q1 * -1, q1)

        # Relative rotation quaternion
        q_rel = self.quaternion_multiply(self.quaternion_conjugate(q0), q1)

        # Convert to axis-angle for angular velocity
        angle = jnp.arccos(jnp.clip(q_rel[..., 0], -1.0, 1.0))
        axis = q_rel[..., 1:] / jnp.linalg.norm(q_rel[..., 1:], axis=-1, keepdims=True)
        omega = angle[..., None] * axis

        # Linear velocity: relative translation rotated back to P0 frame
        t_rel = t1 - t0
        v = self.rotate_vector_by_quaternion(self.quaternion_conjugate(q0), t_rel)

        return jnp.concatenate([omega, v], axis=-1)

    def exponential_map(self, p, v, delta_t):
        # p: [qw, qx, qy, qz, tx, ty, tz]
        # v: twist [omega_x, omega_y, omega_z, v_x, v_y, v_z]
        q, t = p[..., :4], p[..., 4:]
        omega, vel = v[..., :3], v[..., 3:]

        # Rotation part: exponential map for SO(3)
        theta = jnp.linalg.norm(omega, axis=-1)[..., None]
        theta = jnp.where(theta < 1e-8, 0, theta)
        u = omega / theta
        alpha = delta_t * theta
        q_rot = jnp.concatenate([jnp.cos(alpha), jnp.sin(alpha) * u], axis=-1)
        q_new = self.quaternion_multiply(q, q_rot)
        q_new = q_new / jnp.linalg.norm(q_new, axis=-1, keepdims=True)

        # Translation part: integrate velocity
        t_new = t + delta_t * self.rotate_vector_by_quaternion(q, vel)

        return jnp.concatenate([q_new, t_new], axis=-1)

    def interpolant(self, P0, P1, t):
        # Normalize P0 and P1
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        velocity = self.velocity(P0, P1, t=0)
        pose_interp = self.exponential_map(P0, velocity, t)
        return pose_interp

    def tangent_norm(self, v, w):
        # Euclidean norm in twist space
        return jnp.mean(jnp.square(v - w))

    def distance(self, P0, P1):
        # Normalize P0 and P1
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)

        # Extract rotations and translations
        q0, t0 = P0[..., :4], P0[..., 4:]
        q1, t1 = P1[..., :4], P1[..., 4:]

        # Rotational distance (squared angle)
        dots = jnp.sum(q0 * q1, axis=-1)
        dots_adj = jnp.where(dots < 0, -dots, dots)
        rot_dist = jnp.arccos(jnp.clip(dots_adj, -1.0, 1.0))**2

        # Translational distance
        trans_dist = jnp.sum((t0 - t1)**2, axis=-1)

        # Combined distance
        return rot_dist + trans_dist

    def distance_matrix(self, P0, P1):
        # Normalize P0 and P1
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)

        # Extract rotations and translations
        q0, t0 = P0[..., :4], P0[..., 4:]
        q1, t1 = P1[..., :4], P1[..., 4:]

        # Rotational distance matrix
        dots = q0[:, None, :] * q1[None, :, :]
        dots_sum = jnp.sum(dots, axis=-1)
        dots_adj = jnp.where(dots_sum < 0, -dots_sum, dots_sum)
        rot_dist_mat = jnp.arccos(jnp.clip(dots_adj, -1.0, 1.0))**2

        # Translational distance matrix
        trans_dist_mat = jnp.sum((t0[:, None, :] - t1[None, :, :])**2, axis=-1)

        # Combined distance matrix
        return rot_dist_mat + trans_dist_mat
    


if __name__ == "__main__":
    ### SO3 test
    import jax
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    # Create random arrays of sie 10x5x4
    P0 = jax.random.uniform(subkey, (10,5,4))
    key, subkey = jax.random.split(key)
    P1 = jax.random.uniform(subkey, (10,5,4))
    geo_cls = so3()
    P0 = geo_cls.project_to_geometry(P0)
    P1 = geo_cls.project_to_geometry(P1)
    # Check if 0 and 1 are close to endpoints
    assert jnp.allclose(geo_cls.interpolant(P0,P1,0),P0)
    assert jnp.allclose(geo_cls.interpolant(P0,P1,1),P1)

if __name__ == "__main__":
    ### SE3 test
    import jax
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    # Create random arrays of size 10x5x7
    P0 = jax.random.uniform(subkey, (10,5,7))
    key, subkey = jax.random.split(key)
    P1 = jax.random.uniform(subkey, (10,5,7))
    geo_cls = se3()
    P0 = geo_cls.project_to_geometry(P0)
    P1 = geo_cls.project_to_geometry(P1)
    # Check if 0 and 1 are close to endpoints
    assert jnp.allclose(geo_cls.interpolant(P0,P1,0),P0)
    assert jnp.allclose(geo_cls.interpolant(P0,P1,1),P1)

    # Test SE3
    key, subkey = jax.random.split(key)
    P0_se3 = jax.random.uniform(subkey, (5, 7))
    key, subkey = jax.random.split(key)
    P1_se3 = jax.random.uniform(subkey, (5, 7))
    geo_se3 = se3()
    P0_se3 = geo_se3.project_to_geometry(P0_se3)
    P1_se3 = geo_se3.project_to_geometry(P1_se3)

    dist_se3 = geo_se3.distance(P0_se3[0], P1_se3[0])
    dist_mat_se3 = geo_se3.distance_matrix(P0_se3, P1_se3)
    print('SE3 distance shape:', dist_se3.shape)
    print('SE3 distance matrix shape:', dist_mat_se3.shape)
    print('Test passed')