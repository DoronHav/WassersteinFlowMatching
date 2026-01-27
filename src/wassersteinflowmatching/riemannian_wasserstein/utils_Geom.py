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
        return jnp.sum(diff**2, axis=-1)

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

class SO3:
    def project_to_geometry(self, P, use_cpu=False):
        # Normalize to unit quaternion (S^3)
        # P shape: (..., 4)
        if use_cpu:
            norm = np.linalg.norm(P, axis=-1, keepdims=True)
            return np.nan_to_num(P / norm, nan=1.0/np.sqrt(P.shape[-1]))
        norm = jnp.linalg.norm(P, axis=-1, keepdims=True)
        return jnp.nan_to_num(P / norm, nan=1.0/jnp.sqrt(P.shape[-1]))

    def distance(self, P0, P1):
        # Geodesic distance on SO(3) with double cover handling
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Dot product
        dot = jnp.sum(P0 * P1, axis=-1)
        
        # Account for double cover: q and -q are the same rotation
        abs_dot = jnp.abs(dot)
        abs_dot = jnp.clip(abs_dot, -1.0, 1.0)
        
        # Distance is 2 * arccos(|<q1, q2>|)
        # Return squared distance as per other classes
        return (2 * jnp.arccos(abs_dot))**2

    def distance_matrix(self, P0, P1):
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        # Matrix of dot products
        dot_mat = P0 @ P1.T
        
        abs_dot_mat = jnp.abs(dot_mat)
        abs_dot_mat = jnp.clip(abs_dot_mat, -1.0, 1.0)
        
        return (2 * jnp.arccos(abs_dot_mat))**2

    def interpolant(self, P0, P1, t):
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        dot = jnp.sum(P0 * P1, axis=-1)
        
        # Flip P1 if dot < 0 to take shortest path
        sign = jnp.sign(dot)
        sign = jnp.where(sign == 0, 1.0, sign)
        P1 = P1 * sign[..., None]
        
        # Now standard SLERP on S^3
        dot = jnp.abs(dot)
        dot = jnp.clip(dot, -1.0, 1.0)
        
        theta = jnp.arccos(dot)
        sin_theta = jnp.sin(theta)
        
        a = jnp.where(sin_theta < 1e-6, 1.0 - t, jnp.sin((1 - t) * theta) / sin_theta)
        b = jnp.where(sin_theta < 1e-6, t, jnp.sin(t * theta) / sin_theta)
        
        return a[..., None] * P0 + b[..., None] * P1

    def velocity(self, P0, P1, t):
        P0 = self.project_to_geometry(P0)
        P1 = self.project_to_geometry(P1)
        
        dot = jnp.sum(P0 * P1, axis=-1)
        
        sign = jnp.sign(dot)
        sign = jnp.where(sign == 0, 1.0, sign)
        P1 = P1 * sign[..., None]
        
        dot = jnp.abs(dot)
        dot = jnp.clip(dot, -1.0, 1.0)
        
        theta = jnp.arccos(dot)
        sin_theta = jnp.sin(theta)
        
        a = jnp.where(sin_theta < 1e-6, -1.0, -theta * jnp.cos((1 - t) * theta) / sin_theta)
        b = jnp.where(sin_theta < 1e-6, 1.0, theta * jnp.cos(t * theta) / sin_theta)
        
        return a[..., None] * P0 + b[..., None] * P1

    def tangent_norm(self, v, w, p):
        p = self.project_to_geometry(p)
        # Project to tangent space of S^3 at p
        v_tan = v - jnp.sum(v * p, axis=-1, keepdims=True) * p
        w_tan = w - jnp.sum(w * p, axis=-1, keepdims=True) * p
        
        return jnp.mean(jnp.square(v_tan - w_tan))

    def exponential_map(self, p, v, delta_t):
        # Exponential map on S^3
        p = self.project_to_geometry(p)
        
        # Project v to tangent space
        v = v - jnp.sum(v * p, axis=-1, keepdims=True) * p
        
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        
        def small_step():
            step = p + v * delta_t
            return self.project_to_geometry(step)
            
        def general_step():
            theta = v_norm * delta_t
            return jnp.cos(theta) * p + jnp.sin(theta) * (v / (v_norm + 1e-9))
            
        return jnp.where(v_norm < 1e-6, small_step(), general_step())

    def weighted_mean(self, points, weights):
        # Eigenvector method
        weights = weights / (jnp.sum(weights) + 1e-9)
        
        # M = sum w_i q_i q_i^T
        # points: (N, 4)
        M = jnp.einsum('n,ni,nj->ij', weights, points, points)
        
        eigvals, eigvecs = jnp.linalg.eigh(M)
        mean_q = eigvecs[:, -1]
        
        return self.project_to_geometry(mean_q)

class SE3:
    def __init__(self):
        self.so3 = SO3()
        self.euc = euclidean()

    def project_to_geometry(self, P, use_cpu=False):
        # trans is first (:3), rot is second (-3:)
        rot = P[..., 3:]
        trans = P[..., :3]
        rot = self.so3.project_to_geometry(rot, use_cpu=use_cpu)
        if use_cpu:
            return np.concatenate([trans, rot], axis=-1)
        return jnp.concatenate([trans, rot], axis=-1)

    def distance(self, P0, P1):
        rot0, trans0 = P0[..., 3:], P0[..., :3]
        rot1, trans1 = P1[..., 3:], P1[..., :3]
        
        d_rot = self.so3.distance(rot0, rot1)
        d_trans = self.euc.distance(trans0, trans1)
        
        return jnp.sum(d_rot) + d_trans

    def distance_matrix(self, P0, P1):
        # P0: (B1, ..., 7)
        # P1: (B2, ..., 7)
        
        trans0 = P0[..., :3]
        trans1 = P1[..., :3]
        rot0 = P0[..., 3:]
        rot1 = P1[..., 3:]
        
        # Translation distance matrix
        # Expand for broadcasting: (B1, 1, ..., 3) and (1, B2, ..., 3)
        diff_trans = trans0[:, None, ...] - trans1[None, :, ...]
        # Sum over all dimensions starting from axis 2
        d_trans = jnp.sum(diff_trans**2, axis=tuple(range(2, diff_trans.ndim)))
        
        # Rotation distance matrix
        # Dot product: (B1, 1, ..., 4) * (1, B2, ..., 4) -> sum last axis
        dot = jnp.sum(rot0[:, None, ...] * rot1[None, :, ...], axis=-1)
        abs_dot = jnp.clip(jnp.abs(dot), -1.0, 1.0)
        d_rot_sq = (2 * jnp.arccos(abs_dot))**2
        
        # Sum over all dimensions starting from axis 2 (the N dimensions)
        d_rot = jnp.sum(d_rot_sq, axis=tuple(range(2, d_rot_sq.ndim)))
        
        return d_rot + d_trans

    def interpolant(self, P0, P1, t):
        rot0, trans0 = P0[..., 3:], P0[..., :3]
        rot1, trans1 = P1[..., 3:], P1[..., :3]
        
        rot_t = self.so3.interpolant(rot0, rot1, t)
        trans_t = self.euc.interpolant(trans0, trans1, t)
        
        return jnp.concatenate([trans_t, rot_t], axis=-1)

    def velocity(self, P0, P1, t):
        rot0, trans0 = P0[..., 3:], P0[..., :3]
        rot1, trans1 = P1[..., 3:], P1[..., :3]
        
        rot_v = self.so3.velocity(rot0, rot1, t)
        trans_v = self.euc.velocity(trans0, trans1, t)
        
        return jnp.concatenate([trans_v, rot_v], axis=-1)

    def tangent_norm(self, v, w, p):
        p_rot = p[..., 3:]
        v_rot, v_trans = v[..., 3:], v[..., :3]
        w_rot, w_trans = w[..., 3:], w[..., :3]
        
        p_rot = self.so3.project_to_geometry(p_rot)
        v_rot_tan = v_rot - jnp.sum(v_rot * p_rot, axis=-1, keepdims=True) * p_rot
        w_rot_tan = w_rot - jnp.sum(w_rot * p_rot, axis=-1, keepdims=True) * p_rot
        
        diff_rot = v_rot_tan - w_rot_tan
        diff_trans = v_trans - w_trans
        
        diff = jnp.concatenate([diff_trans, diff_rot], axis=-1)
        return jnp.mean(jnp.square(diff))

    def exponential_map(self, p, v, delta_t):
        p_rot, p_trans = p[..., 3:], p[..., :3]
        v_rot, v_trans = v[..., 3:], v[..., :3]
        
        new_rot = self.so3.exponential_map(p_rot, v_rot, delta_t)
        new_trans = self.euc.exponential_map(p_trans, v_trans, delta_t)
        
        return jnp.concatenate([new_trans, new_rot], axis=-1)

    def weighted_mean(self, points, weights):
        rot = points[..., 3:]
        trans = points[..., :3]
        
        mean_rot = self.so3.weighted_mean(rot, weights)
        mean_trans = self.euc.weighted_mean(trans, weights)
        
        return jnp.concatenate([mean_trans, mean_rot], axis=-1)

class SE3_n(SE3):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def project_to_geometry(self, P, use_cpu=False):
        # P: (..., 7*n)
        shape = P.shape
        P = P.reshape(shape[:-1] + (self.n, 7))
        trans = P[..., :3]
        rot = P[..., 3:]
        
        rot = self.so3.project_to_geometry(rot, use_cpu=use_cpu)
        
        if use_cpu:
            res = np.concatenate([trans, rot], axis=-1)
        else:
            res = jnp.concatenate([trans, rot], axis=-1)
        return res.reshape(shape)

    def distance(self, P0, P1, mask0=None, mask1=None):

        if mask0 is None:
            mask0 = jnp.ones(P0.shape[:-1] + (self.n,)) 
        if mask1 is None:
            mask1 = jnp.ones(P1.shape[:-1] + (self.n,))

        mask_combined = mask0 * mask1

        P0 = P0.reshape(P0.shape[:-1] + (self.n, 7))
        P1 = P1.reshape(P1.shape[:-1] + (self.n, 7))
        
        trans0, rot0 = P0[..., :3], P0[..., 3:]
        trans1, rot1 = P1[..., :3], P1[..., 3:]
        
        d_trans = jnp.sum((trans0 - trans1)**2, axis=-1)
        d_rot = self.so3.distance(rot0, rot1)
        
        dist_comp = d_trans + d_rot
        
        if mask0 is not None:
            return jnp.sum(dist_comp * mask_combined, axis=-1)
            
        return jnp.sum(dist_comp, axis=-1)

    def distance_matrix(self, P0, P1, mask0=None, mask1=None):

        if mask0 is None:
            mask0 = jnp.ones(P0.shape[:-1] + (self.n,)) 
        if mask1 is None:
            mask1 = jnp.ones(P1.shape[:-1] + (self.n,))

        mask_outer = mask0[:, None, :] * mask1[None, :, :]

        P0 = P0.reshape(P0.shape[:-1] + (self.n, 7))
        P1 = P1.reshape(P1.shape[:-1] + (self.n, 7))
        
        trans0 = P0[..., :3] # nx, n_residues, 3
        trans1 = P1[..., :3] # ny, n_residues, 3
        rot0 = P0[..., 3:] # nx, n_residues, 4
        rot1 = P1[..., 3:] # ny, n_residues, 4
        
        # Translation
        diff_trans = trans0[:, None, ...] - trans1[None, :, ...]
        d_trans = jnp.sum(diff_trans**2, axis=-1) #n_x, n_y, n_residues
        
        # Rotation
        dot = jnp.sum(rot0[:, None, ...] * rot1[None, :, ...], axis=-1) # n_x, n_y, n_residues
        abs_dot = jnp.clip(jnp.abs(dot), -1.0, 1.0) # n_x, n_y, n_residues
        d_rot = (2 * jnp.arccos(abs_dot))**2 # n_x, n_y, n_residues
        
        dist_comp = d_trans + d_rot # n_x, n_y, n_residues
        dist_comp = dist_comp * mask_outer
        
        return jnp.sum(dist_comp, axis=-1)

    def interpolant(self, P0, P1, t):
        shape = P0.shape
        P0 = P0.reshape(shape[:-1] + (self.n, 7))
        P1 = P1.reshape(shape[:-1] + (self.n, 7))
        
        trans0, rot0 = P0[..., :3], P0[..., 3:]
        trans1, rot1 = P1[..., :3], P1[..., 3:]
        
        trans_t = self.euc.interpolant(trans0, trans1, t)
        rot_t = self.so3.interpolant(rot0, rot1, t)
        
        res = jnp.concatenate([trans_t, rot_t], axis=-1)
        return res.reshape(shape)

    def velocity(self, P0, P1, t):
        shape = P0.shape
        P0 = P0.reshape(shape[:-1] + (self.n, 7))
        P1 = P1.reshape(shape[:-1] + (self.n, 7))
        
        trans0, rot0 = P0[..., :3], P0[..., 3:]
        trans1, rot1 = P1[..., :3], P1[..., 3:]
        
        trans_v = self.euc.velocity(trans0, trans1, t)
        rot_v = self.so3.velocity(rot0, rot1, t)
        
        res = jnp.concatenate([trans_v, rot_v], axis=-1)
        return res.reshape(shape)

    def tangent_norm(self, v, w, p, mask_p=None):
        shape = p.shape
        p = p.reshape(shape[:-1] + (self.n, 7))
        v = v.reshape(shape[:-1] + (self.n, 7))
        w = w.reshape(shape[:-1] + (self.n, 7))
        
        p_rot = p[..., 3:] # n_x, n_residues, 4
        v_rot, v_trans = v[..., 3:], v[..., :3] # n_x, n_residues, 4/3
        w_rot, w_trans = w[..., 3:], w[..., :3] # n_x, n_residues, 4/3
        
        p_rot = self.so3.project_to_geometry(p_rot)
        
        v_rot_tan = v_rot - jnp.sum(v_rot * p_rot, axis=-1, keepdims=True) * p_rot
        w_rot_tan = w_rot - jnp.sum(w_rot * p_rot, axis=-1, keepdims=True) * p_rot
        
        diff_rot = v_rot_tan - w_rot_tan
        diff_trans = v_trans - w_trans
        
        diff = jnp.concatenate([diff_trans, diff_rot], axis=-1) # n_x, n_residues, 7
        sq_norm = jnp.square(diff) # n_x, n_residues, 7
        
        # mask_p is of shape n_x, n_residues
        if mask_p is not None:
            sq_norm = sq_norm * mask_p[..., None]
            return jnp.mean(sq_norm)
            
        return jnp.mean(sq_norm)

    def exponential_map(self, p, v, delta_t):
        shape = p.shape
        p = p.reshape(shape[:-1] + (self.n, 7))
        v = v.reshape(shape[:-1] + (self.n, 7))
        
        p_trans, p_rot = p[..., :3], p[..., 3:]
        v_trans, v_rot = v[..., :3], v[..., 3:]
        
        new_trans = self.euc.exponential_map(p_trans, v_trans, delta_t)
        new_rot = self.so3.exponential_map(p_rot, v_rot, delta_t)
        
        res = jnp.concatenate([new_trans, new_rot], axis=-1)
        return res.reshape(shape)

    def weighted_mean(self, points, weights, mask=None):
        N = points.shape[0]
        points = points.reshape(N, self.n, 7)
        
        if mask is not None:
            mask = mask.reshape(N, self.n, 7)[..., 0]
            w_comp = weights[:, None] * mask
            w_sum = jnp.sum(w_comp, axis=0) + 1e-9
            w_norm = w_comp / w_sum[None, :]
            
            trans = points[..., :3]
            mean_trans = jnp.sum(trans * w_norm[..., None], axis=0)
            
            rot = points[..., 3:]
            M = jnp.einsum('nk,nki,nkj->kij', w_norm, rot, rot)
            
            eigvals, eigvecs = jnp.linalg.eigh(M)
            mean_rot = eigvecs[..., -1]
            mean_rot = self.so3.project_to_geometry(mean_rot)
            
            res_val = jnp.concatenate([mean_trans, mean_rot], axis=-1)
            return res_val.flatten()
        
        trans = points[..., :3]
        rot = points[..., 3:]
        
        w_sum = jnp.sum(weights) + 1e-9
        w_expanded = weights[:, None, None]
        mean_trans = jnp.sum(trans * w_expanded, axis=0) / w_sum
        
        weights_norm = weights / w_sum
        M = jnp.einsum('n,nki,nkj->kij', weights_norm, rot, rot)
        
        eigvals, eigvecs = jnp.linalg.eigh(M)
        mean_rot = eigvecs[..., -1]
        
        mean_rot = self.so3.project_to_geometry(mean_rot)
        
        res = jnp.concatenate([mean_trans, mean_rot], axis=-1)
        return res.reshape(self.n * 7)

class Euclidean_n:
    def __init__(self, n):
        self.n = n
        self.euc = euclidean()

    def project_to_geometry(self, P, use_cpu=False):
        return P

    def distance(self, P0, P1, mask0=None, mask1=None):

        if mask0 is None:
            mask0 = jnp.ones(P0.shape[:-1] + (self.n,)) 
        if mask1 is None:
            mask1 = jnp.ones(P1.shape[:-1] + (self.n,))

        mask_combined = mask0 * mask1

        P0 = P0.reshape(P0.shape[:-1] + (self.n, 7))
        P1 = P1.reshape(P1.shape[:-1] + (self.n, 7))
        
        dist_comp = jnp.sum((P0 - P1)**2, axis=-1)
        
        if mask0 is not None:
            return jnp.sum(dist_comp * mask_combined, axis=-1)
            
        return jnp.sum(dist_comp, axis=-1)

    def distance_matrix(self, P0, P1, mask0=None, mask1=None):

        if mask0 is None:
            mask0 = jnp.ones(P0.shape[:-1] + (self.n,)) 
        if mask1 is None:
            mask1 = jnp.ones(P1.shape[:-1] + (self.n,))

        mask_outer = mask0[:, None, :] * mask1[None, :, :]

        P0 = P0.reshape(P0.shape[:-1] + (self.n, 7))
        P1 = P1.reshape(P1.shape[:-1] + (self.n, 7))
        
        diff = P0[:, None, ...] - P1[None, :, ...]
        dist_comp = jnp.sum(diff**2, axis=-1)
        
        dist_comp = dist_comp * mask_outer
        
        return jnp.sum(dist_comp, axis=-1)

    def interpolant(self, P0, P1, t):
        return (1 - t) * P0 + t * P1

    def velocity(self, P0, P1, t):
        return P1 - P0

    def tangent_norm(self, v, w, p, mask_p=None):
        shape = p.shape
        v = v.reshape(shape[:-1] + (self.n, 7))
        w = w.reshape(shape[:-1] + (self.n, 7))

        diff = v - w
        sq_norm = jnp.square(diff)
        
        if mask_p is not None:
            sq_norm = sq_norm * mask_p[..., None]
            
        return jnp.mean(sq_norm)

    def exponential_map(self, p, v, delta_t):
        return p + v * delta_t

    def weighted_mean(self, points, weights, mask=None):
        N = points.shape[0]
        points = points.reshape(N, self.n, 7)
        
        if mask is not None:
            mask = mask.reshape(N, self.n, 7)[..., 0]
            w_comp = weights[:, None] * mask
            w_sum = jnp.sum(w_comp, axis=0) + 1e-9
            w_norm = w_comp / w_sum[None, :]
            
            mean_val = jnp.sum(points * w_norm[..., None], axis=0)
            return mean_val.flatten()
        
        w_sum = jnp.sum(weights) + 1e-9
        w_expanded = weights[:, None, None]
        mean_val = jnp.sum(points * w_expanded, axis=0) / w_sum
        return mean_val.reshape(self.n * 7)

        