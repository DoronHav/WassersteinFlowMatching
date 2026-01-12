"""
SE(3)-Equivariant Attention Network for Protein Trajectory Data

Input shape: (B, N_timesteps, N_residues * 7)
    - Each timestep is a protein conformation with N_residues frames
    - Each frame is [qw, qx, qy, qz, tx, ty, tz]

Equivariance: If all frames are globally rotated/translated, output velocities
transform accordingly.

Architecture:
    1. Reshape to (B, N_timesteps, N_residues, 7)
    2. Compute SE(3)-invariant features per timestep (pooled over residues)
    3. Standard attention over N_timesteps
    4. Predict equivariant velocity per residue
    5. Reshape back to (B, N_timesteps, N_residues * 7)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, NamedTuple


# =============================================================================
# Quaternion Operations
# =============================================================================

def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product: q1 * q2. Convention: [w, x, y, z]"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=-1)


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    return q * jnp.array([1., -1., -1., -1.])


def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def quat_to_matrix(q: jnp.ndarray) -> jnp.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    return jnp.stack([
        jnp.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        jnp.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        jnp.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1),
    ], axis=-2)


def rotate_vector(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    R = quat_to_matrix(q)
    return jnp.einsum('...ij,...j->...i', R, v)


# =============================================================================
# Frame Utilities
# =============================================================================

class Frames(NamedTuple):
    quat: jnp.ndarray   # (..., 4)
    trans: jnp.ndarray  # (..., 3)


def tensor_to_frames(x: jnp.ndarray) -> Frames:
    """x: (..., 7) -> Frames"""
    return Frames(quat=quat_normalize(x[..., :4]), trans=x[..., 4:])


def rbf_encode(x: jnp.ndarray, num_rbf: int = 16, max_val: float = 20.0) -> jnp.ndarray:
    centers = jnp.linspace(0, max_val, num_rbf)
    return jnp.exp(-0.5 * (x[..., None] - centers) ** 2)


# =============================================================================
# Per-Timestep Invariant Feature Extraction
# =============================================================================

def compute_timestep_invariant_features(
    frames: Frames,
    num_rbf: int = 16,
) -> jnp.ndarray:
    """
    Compute SE(3)-invariant features for each timestep by pooling over residues.
    
    frames.quat: (B, N_timesteps, N_residues, 4)
    frames.trans: (B, N_timesteps, N_residues, 3)
    
    Returns: (B, N_timesteps, feature_dim)
    
    All features here are INVARIANT under global SE(3) transformations.
    """
    B, T, R, _ = frames.trans.shape
    
    # === Pairwise features within each timestep ===
    # trans: (B, T, R, 3)
    t_i = frames.trans[:, :, :, None, :]  # (B, T, R, 1, 3)
    t_j = frames.trans[:, :, None, :, :]  # (B, T, 1, R, 3)
    t_diff = t_j - t_i  # (B, T, R, R, 3)
    
    # Pairwise distances (invariant)
    distances = jnp.linalg.norm(t_diff + 1e-8, axis=-1)  # (B, T, R, R)
    
    # RBF encode distances
    dist_rbf = rbf_encode(distances, num_rbf)  # (B, T, R, R, num_rbf)
    
    # Local direction: R_i^T @ (t_j - t_i) - invariant!
    q_i_inv = quat_conjugate(frames.quat)[:, :, :, None, :]  # (B, T, R, 1, 4)
    local_dir = rotate_vector(
        jnp.broadcast_to(q_i_inv, (B, T, R, R, 4)),
        t_diff
    )  # (B, T, R, R, 3)
    
    # Relative quaternion: q_i^* @ q_j - invariant!
    q_j = frames.quat[:, :, None, :, :]  # (B, T, 1, R, 4)
    rel_quat = quat_multiply(
        jnp.broadcast_to(q_i_inv, (B, T, R, R, 4)),
        jnp.broadcast_to(q_j, (B, T, R, R, 4))
    )  # (B, T, R, R, 4)
    
    # === Pool over residue pairs to get per-timestep features ===
    # Mean pool over both residue dimensions
    dist_feat = jnp.mean(dist_rbf, axis=(2, 3))  # (B, T, num_rbf)
    
    # Local direction statistics (invariant because in local frames)
    local_dir_mean = jnp.mean(local_dir, axis=(2, 3))  # (B, T, 3)
    local_dir_std = jnp.std(local_dir, axis=(2, 3))  # (B, T, 3)
    
    # Relative quaternion statistics (invariant!)
    # The scalar part w = cos(theta/2) of relative quaternion
    rel_quat_w_mean = jnp.mean(rel_quat[..., 0], axis=(2, 3))[..., None]  # (B, T, 1)
    rel_quat_w_std = jnp.std(rel_quat[..., 0], axis=(2, 3))[..., None]  # (B, T, 1)
    # Vector part magnitude (related to rotation angle)
    rel_quat_xyz_norm = jnp.linalg.norm(rel_quat[..., 1:], axis=-1)  # (B, T, R, R)
    rel_quat_xyz_mean = jnp.mean(rel_quat_xyz_norm, axis=(2, 3))[..., None]  # (B, T, 1)
    
    # === Single-residue invariant features ===
    # Radius of gyration (invariant under SE(3))
    centroid = jnp.mean(frames.trans, axis=2, keepdims=True)  # (B, T, 1, 3)
    rel_trans = frames.trans - centroid  # (B, T, R, 3)
    rg_sq = jnp.mean(jnp.sum(rel_trans ** 2, axis=-1), axis=2, keepdims=True)  # (B, T, 1)
    rg = jnp.sqrt(rg_sq + 1e-8)  # (B, T, 1)
    
    # Spread in each local axis (invariant)
    local_spread = jnp.std(rel_trans, axis=2)  # (B, T, 3) - this is NOT invariant
    # Fix: compute spread in local frame of first residue (or centroid orientation)
    # Actually, let's use distance-based spread which is invariant
    dist_mean = jnp.mean(distances, axis=(2, 3))[..., None]  # (B, T, 1)
    dist_std = jnp.std(distances, axis=(2, 3))[..., None]  # (B, T, 1)
    dist_max = jnp.max(distances, axis=(2, 3))[..., None]  # (B, T, 1)
    
    # Concatenate all INVARIANT features
    features = jnp.concatenate([
        dist_feat,          # (B, T, num_rbf) - pairwise distance distribution
        local_dir_mean,     # (B, T, 3) - mean local direction
        local_dir_std,      # (B, T, 3) - local direction spread
        rel_quat_w_mean,    # (B, T, 1) - mean relative rotation angle
        rel_quat_w_std,     # (B, T, 1) - spread of relative rotations
        rel_quat_xyz_mean,  # (B, T, 1) - mean rotation magnitude
        rg,                 # (B, T, 1) - radius of gyration
        dist_mean,          # (B, T, 1) - mean pairwise distance
        dist_std,           # (B, T, 1) - distance spread
        dist_max,           # (B, T, 1) - max extent
    ], axis=-1)
    
    return features  # (B, T, num_rbf + 14)


# =============================================================================
# Feed Forward
# =============================================================================

class FeedForward(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, inputs, deterministic: bool = True, dropout_rng=None):
        x = nn.Dense(features=self.config.mlp_hidden_dim)(inputs)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic, rng=dropout_rng)
        x = nn.leaky_relu(x)
        output = nn.Dense(inputs.shape[-1])(x)
        return output


# =============================================================================
# Encoder Block (attention over timesteps)
# =============================================================================

class EncoderBlock(nn.Module):
    """Standard transformer encoder block operating over N_timesteps."""
    
    config: dict

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,  # (B, N_timesteps, embedding_dim)
        masks: Optional[jnp.ndarray] = None,  # (B, N_timesteps)
        dropout_rng: Optional[jnp.ndarray] = None,
        t_emb: Optional[jnp.ndarray] = None,  # (B, embedding_dim)
        c_emb: Optional[jnp.ndarray] = None,  # (B, embedding_dim)
        deterministic: bool = True,
    ) -> jnp.ndarray:
        
        num_heads = self.config.num_heads
        dropout_rate = self.config.dropout_rate

        # Conditioning
        conditioning = jnp.zeros_like(inputs)
        if t_emb is not None:
            conditioning += t_emb[:, None, :]
        if c_emb is not None:
            conditioning += c_emb[:, None, :]
        
        conditioned_inputs = inputs + conditioning
        attn_mask = masks[:, None, None, :] if masks is not None else None

        attn_rng, ff_rng = jax.random.split(dropout_rng) if dropout_rng is not None else (None, None)
        
        # Pre-norm attention
        normed_inputs = nn.LayerNorm()(conditioned_inputs)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )(
            normed_inputs,
            mask=attn_mask,
            deterministic=deterministic,
            dropout_rng=attn_rng
        )
        
        x = inputs + attn_output
        
        # Pre-norm FFN
        normed_x = nn.LayerNorm()(x)
        ff_output = FeedForward(config=self.config)(normed_x, deterministic=deterministic, dropout_rng=ff_rng)
        output = x + ff_output
        
        return output


# =============================================================================
# Equivariant Output Head
# =============================================================================

class EquivariantOutputHead(nn.Module):
    """
    Predicts SE(3)-equivariant velocities for each residue.
    
    Takes scalar features and frames, outputs velocity in (B, T, R, 7) format.
    """
    
    @nn.compact
    def __call__(
        self,
        scalar_features: jnp.ndarray,  # (B, T, embedding_dim)
        frames: Frames,                 # quat: (B, T, R, 4), trans: (B, T, R, 3)
    ) -> jnp.ndarray:
        
        B, T, R, _ = frames.trans.shape
        d = scalar_features.shape[-1]
        
        # Expand scalar features to per-residue
        # (B, T, d) -> (B, T, R, d)
        x = jnp.broadcast_to(scalar_features[:, :, None, :], (B, T, R, d))
        
        # Add per-residue positional info (relative to centroid, in local frame)
        centroid = jnp.mean(frames.trans, axis=2, keepdims=True)  # (B, T, 1, 3)
        rel_pos = frames.trans - centroid  # (B, T, R, 3)
        
        # Transform to local frame (invariant)
        local_rel_pos = rotate_vector(quat_conjugate(frames.quat), rel_pos)  # (B, T, R, 3)
        
        # Concatenate and project
        x = jnp.concatenate([x, local_rel_pos], axis=-1)  # (B, T, R, d+3)
        x = nn.Dense(d)(x)
        x = nn.gelu(x)
        x = nn.Dense(d // 2)(x)
        x = nn.gelu(x)
        
        # Translation velocity: predict in local frame, rotate to global
        trans_vel_local = nn.Dense(3)(x)  # (B, T, R, 3)
        trans_vel = rotate_vector(frames.quat, trans_vel_local)  # Equivariant!
        
        # Rotation velocity: predict axis-angle in local frame
        rot_vel_local = nn.Dense(3)(x) * 0.1  # (B, T, R, 3)
        
        # Convert to quaternion velocity: dq/dt = 0.5 * q * [0, omega]
        omega_quat = jnp.concatenate([
            jnp.zeros((B, T, R, 1)),
            rot_vel_local
        ], axis=-1)  # (B, T, R, 4)
        quat_vel = 0.5 * quat_multiply(frames.quat, omega_quat)  # (B, T, R, 4)
        
        # Combine: (B, T, R, 7)
        output = jnp.concatenate([quat_vel, trans_vel], axis=-1)
        
        return output


# =============================================================================
# Main Model
# =============================================================================

class SE3AttentionNN(nn.Module):
    """
    SE(3)-Equivariant attention network for protein trajectory flow matching.
    
    Input: (B, N_timesteps, N_residues * 7) - flattened trajectory
    Output: (B, N_timesteps, N_residues * 7) - velocity field
    
    The model:
    1. Reshapes to (B, T, R, 7) to access frame structure
    2. Computes SE(3)-invariant features per timestep
    3. Runs attention over timesteps
    4. Outputs equivariant velocities per residue
    5. Reshapes back to (B, T, R * 7)
    """
    
    config: dict

    @nn.compact
    def __call__(
        self,
        point_cloud: jnp.ndarray,  # (B, N_timesteps, N_residues * 7)
        t: jnp.ndarray,            # (B,)
        masks: Optional[jnp.ndarray] = None,  # (B, N_timesteps)
        conditioning: Optional[jnp.ndarray] = None,
        is_null_conditioning: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        dropout_rng: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        
        config = self.config
        embedding_dim = config.embedding_dim
        num_layers = config.num_layers
        
        B, T, flat_dim = point_cloud.shape
        assert flat_dim % 7 == 0, f"Last dimension must be divisible by 7, got {flat_dim}"
        R = flat_dim // 7  # Infer N_residues
        
        embedding_dim = config.num_heads * (embedding_dim // config.num_heads)
        
        if masks is None:
            masks = jnp.ones((B, T))
        
        # === Reshape to access frame structure ===
        frames_tensor = point_cloud.reshape(B, T, R, 7)
        frames = tensor_to_frames(frames_tensor)  # quat: (B,T,R,4), trans: (B,T,R,3)
        
        # === Compute invariant features per timestep ===
        invariant_feat = compute_timestep_invariant_features(frames)  # (B, T, feat_dim)
        
        # Project to embedding dim
        x = nn.Dense(embedding_dim)(invariant_feat)  # (B, T, embedding_dim)
        
        # === Time embedding ===
        freqs = jnp.arange(embedding_dim // 2) * (2.0 * jnp.pi / embedding_dim)
        t_freq = freqs[None, :] * t[:, None]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis=-1)
        t_emb = nn.Dense(features=embedding_dim)(t_four)
        
        # === Conditioning ===
        c_emb = None
        if conditioning is not None:
            if is_null_conditioning is None:
                is_null_conditioning = jnp.zeros(B, dtype=bool)
            
            if config.normalized_condition:
                c_emb = nn.Dense(features=embedding_dim)(conditioning)
                norm = jnp.linalg.norm(c_emb, axis=-1, keepdims=True)
                c_emb = c_emb / (norm + 1e-8)
                c_emb = jnp.where(is_null_conditioning[:, None], jnp.zeros_like(c_emb), c_emb)
            else:
                conditioning = jnp.where(
                    is_null_conditioning[:, None],
                    jnp.zeros_like(conditioning),
                    conditioning
                )
                c_emb = nn.Dense(features=embedding_dim)(conditioning)
        
        # === Transformer layers (attention over timesteps) ===
        for _ in range(num_layers):
            if dropout_rng is not None:
                dropout_rng, layer_rng = jax.random.split(dropout_rng)
            else:
                layer_rng = None
            
            x = EncoderBlock(config=config)(
                x,
                masks=masks,
                dropout_rng=layer_rng,
                t_emb=t_emb,
                c_emb=c_emb,
                deterministic=deterministic,
            )
        
        # === Equivariant output head ===
        output = EquivariantOutputHead()(x, frames)  # (B, T, R, 7)
        
        # === Reshape back to flat ===
        output = output.reshape(B, T, R * 7)
        
        # Apply mask
        output = output * masks[..., None]
        
        return output