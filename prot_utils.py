"""
PDB to SE(3) Frames Converter

Converts protein backbone coordinates to SE(3) frames represented as
(quaternion, translation) tuples of shape (N_res, 7).

The frame construction follows the standard convention:
- Origin: Cα position
- x-axis: Cα → C direction (normalized)
- y-axis: in the Cα-N-C plane, orthogonal to x
- z-axis: x × y (completes right-handed frame)

The quaternion uses (w, x, y, z) convention where w is the scalar part.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Optional
from pathlib import Path
import mdtraj as md

class BackboneFrames(NamedTuple):
    """SE(3) frames for protein backbone residues."""
    quaternions: jnp.ndarray  # (N_res, 4) - [w, x, y, z] convention
    translations: jnp.ndarray  # (N_res, 3) - Cα positions
    frames: jnp.ndarray  # (N_res, 7) - concatenated [quat, trans]
    rotation_matrices: jnp.ndarray  # (N_res, 3, 3) - for verification
    mask: jnp.ndarray  # (N_res,) - valid residues

    def __repr__(self) -> str:
        return (f"BackboneFrames(num_residues={len(self.mask)}, valid_residues={self.mask.sum()})")

def parse_pdb(pdb_path: str) -> dict[str, np.ndarray]:
    """
    Parse PDB file and extract backbone atom coordinates.
    
    Returns dict with 'N', 'CA', 'C' keys mapping to (N_res, 3) arrays.
    Missing atoms are filled with NaN.
    """
    backbone_atoms = {'N': {}, 'CA': {}, 'C': {}}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            
            atom_name = line[12:16].strip()
            if atom_name not in backbone_atoms:
                continue
            
            res_seq = int(line[22:26].strip())
            chain_id = line[21]
            # Use (chain, res_seq) as key to handle multi-chain
            key = (chain_id, res_seq)
            
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            backbone_atoms[atom_name][key] = np.array([x, y, z])
    
    # Get all residue keys and sort them
    all_keys = set()
    for atom_dict in backbone_atoms.values():
        all_keys.update(atom_dict.keys())
    sorted_keys = sorted(all_keys)
    
    n_res = len(sorted_keys)
    coords = {atom: np.full((n_res, 3), np.nan) for atom in backbone_atoms}
    
    for i, key in enumerate(sorted_keys):
        for atom in backbone_atoms:
            if key in backbone_atoms[atom]:
                coords[atom][i] = backbone_atoms[atom][key]
    
    return coords


def rotation_matrix_to_quaternion(R: jnp.ndarray) -> jnp.ndarray:
    """
    Convert rotation matrix to quaternion using Shepperd's method.
    
    This is numerically stable for all rotations.
    
    Args:
        R: (..., 3, 3) rotation matrices
    
    Returns:
        q: (..., 4) quaternions in [w, x, y, z] format
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    # Shepperd's method: find the largest diagonal element to avoid
    # numerical instability near singularities
    
    trace = jnp.trace(R, axis1=-2, axis2=-1)
    
    # Four possible cases based on which element is largest
    # Case 0: w is largest (trace > 0)
    # Case 1: x is largest (R[0,0] largest diagonal)
    # Case 2: y is largest (R[1,1] largest diagonal)  
    # Case 3: z is largest (R[2,2] largest diagonal)
    
    def case_w(R, trace):
        s = 0.5 / jnp.sqrt(1.0 + trace)
        w = 0.25 / s
        x = (R[..., 2, 1] - R[..., 1, 2]) * s
        y = (R[..., 0, 2] - R[..., 2, 0]) * s
        z = (R[..., 1, 0] - R[..., 0, 1]) * s
        return jnp.stack([w, x, y, z], axis=-1)
    
    def case_x(R, trace):
        s = 2.0 * jnp.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])
        w = (R[..., 2, 1] - R[..., 1, 2]) / s
        x = 0.25 * s
        y = (R[..., 0, 1] + R[..., 1, 0]) / s
        z = (R[..., 0, 2] + R[..., 2, 0]) / s
        return jnp.stack([w, x, y, z], axis=-1)
    
    def case_y(R, trace):
        s = 2.0 * jnp.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2])
        w = (R[..., 0, 2] - R[..., 2, 0]) / s
        x = (R[..., 0, 1] + R[..., 1, 0]) / s
        y = 0.25 * s
        z = (R[..., 1, 2] + R[..., 2, 1]) / s
        return jnp.stack([w, x, y, z], axis=-1)
    
    def case_z(R, trace):
        s = 2.0 * jnp.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1])
        w = (R[..., 1, 0] - R[..., 0, 1]) / s
        x = (R[..., 0, 2] + R[..., 2, 0]) / s
        y = (R[..., 1, 2] + R[..., 2, 1]) / s
        z = 0.25 * s
        return jnp.stack([w, x, y, z], axis=-1)
    
    # Compute all cases
    q_w = case_w(R, trace)
    q_x = case_x(R, trace)
    q_y = case_y(R, trace)
    q_z = case_z(R, trace)
    
    # Select based on which is most stable
    # Use trace and diagonal elements to decide
    diag = jnp.diagonal(R, axis1=-2, axis2=-1)  # (..., 3)
    
    decisions = jnp.stack([
        trace,
        R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2],
        R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2],
        R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1],
    ], axis=-1)
    
    choice = jnp.argmax(decisions, axis=-1)
    
    # Stack all quaternions and select
    all_q = jnp.stack([q_w, q_x, q_y, q_z], axis=1)  # (batch, 4, 4)
    q = jnp.take_along_axis(all_q, choice[:, None, None], axis=1).squeeze(1)
    
    # Normalize to ensure unit quaternion
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    
    # Ensure w >= 0 for canonical representation
    q = jnp.where(q[..., :1] < 0, -q, q)
    
    return q.reshape(*batch_shape, 4)


def quaternion_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: (..., 4) quaternions in [w, x, y, z] format
    
    Returns:
        R: (..., 3, 3) rotation matrices
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Compute rotation matrix elements
    R = jnp.stack([
        jnp.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], axis=-1),
        jnp.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], axis=-1),
        jnp.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], axis=-1),
    ], axis=-2)
    
    return R


def construct_backbone_frames(
    n_coords: jnp.ndarray,
    ca_coords: jnp.ndarray,
    c_coords: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Construct local coordinate frames from backbone atoms.
    
    Convention (following AlphaFold/FrameDiff):
    - Origin: Cα
    - x-axis: Cα → C (normalized)
    - z-axis: (Cα → N) × (Cα → C) (normalized) - perpendicular to backbone plane
    - y-axis: z × x (completes right-handed system)
    
    Args:
        n_coords: (N, 3) N atom positions
        ca_coords: (N, 3) Cα atom positions
        c_coords: (N, 3) C atom positions
    
    Returns:
        rotation_matrices: (N, 3, 3) rotation matrices
        translations: (N, 3) Cα positions (frame origins)
        mask: (N,) boolean mask for valid frames
    """
    # Vectors from Cα
    v1 = c_coords - ca_coords   # Cα → C
    v2 = n_coords - ca_coords   # Cα → N
    
    # Check for valid residues (no NaN coordinates)
    valid = ~(jnp.isnan(v1).any(axis=-1) | jnp.isnan(v2).any(axis=-1))
    
    # x-axis: along Cα → C
    x_axis = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    
    # z-axis: perpendicular to the N-Cα-C plane
    z_axis = jnp.cross(v2, v1)
    z_axis = z_axis / (jnp.linalg.norm(z_axis, axis=-1, keepdims=True) + 1e-8)
    
    # y-axis: completes right-handed system
    y_axis = jnp.cross(z_axis, x_axis)
    y_axis = y_axis / (jnp.linalg.norm(y_axis, axis=-1, keepdims=True) + 1e-8)
    
    # Rotation matrix: columns are the local axes expressed in global frame
    # R @ [1,0,0] = x_axis, etc.
    rotation_matrices = jnp.stack([x_axis, y_axis, z_axis], axis=-1)
    
    # Translation is simply the Cα position
    translations = ca_coords
    
    return rotation_matrices, translations, valid


def pdb_to_se3_frames(pdb_path: str) -> BackboneFrames:
    """
    Main function: Convert PDB file to SE(3) frames.
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        BackboneFrames namedtuple with:
            - quaternions: (N_res, 4) in [w, x, y, z] format
            - translations: (N_res, 3) Cα positions
            - frames: (N_res, 7) concatenated representation
            - rotation_matrices: (N_res, 3, 3) for verification
            - mask: (N_res,) validity mask
    """
    # Parse PDB
    coords = parse_pdb(pdb_path)
    
    # Convert to JAX arrays
    n_coords = jnp.array(coords['N'])
    ca_coords = jnp.array(coords['CA'])
    c_coords = jnp.array(coords['C'])
    
    # Construct frames
    rotation_matrices, translations, mask = construct_backbone_frames(
        n_coords, ca_coords, c_coords
    )
    
    # Convert to quaternions
    quaternions = rotation_matrix_to_quaternion(rotation_matrices)
    
    # Handle invalid residues: set to identity quaternion
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    quaternions = jnp.where(
        mask[:, None],
        quaternions,
        identity_quat
    )
    
    # Concatenate to (N_res, 7) representation
    frames = jnp.concatenate([quaternions, translations], axis=-1)
    
    return BackboneFrames(
        quaternions=quaternions,
        translations=translations,
        frames=frames,
        rotation_matrices=rotation_matrices,
        mask=mask,
    )


# ============================================================================
# Utility functions for working with SE(3) frames
# ============================================================================

def compose_se3(q1: jnp.ndarray, t1: jnp.ndarray, 
                q2: jnp.ndarray, t2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compose two SE(3) transformations: T1 ∘ T2
    
    (R1, t1) ∘ (R2, t2) = (R1 @ R2, R1 @ t2 + t1)
    """
    # Quaternion multiplication for rotation composition
    q_composed = quaternion_multiply(q1, q2)
    
    # Transform t2 by R1, then add t1
    R1 = quaternion_to_rotation_matrix(q1)
    t_composed = jnp.einsum('...ij,...j->...i', R1, t2) + t1
    
    return q_composed, t_composed


def quaternion_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply two quaternions: q1 * q2
    
    Args:
        q1, q2: (..., 4) quaternions in [w, x, y, z] format
    
    Returns:
        q: (..., 4) product quaternion
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return jnp.stack([w, x, y, z], axis=-1)


def inverse_se3(q: jnp.ndarray, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute inverse of SE(3) transformation.
    
    (R, t)^{-1} = (R^T, -R^T @ t)
    """
    # Quaternion inverse (conjugate for unit quaternions)
    q_inv = q.at[..., 1:].multiply(-1)
    
    # Inverse translation
    R_inv = quaternion_to_rotation_matrix(q_inv)
    t_inv = -jnp.einsum('...ij,...j->...i', R_inv, t)
    
    return q_inv, t_inv


def apply_se3_to_points(q: jnp.ndarray, t: jnp.ndarray, 
                        points: jnp.ndarray) -> jnp.ndarray:
    """
    Apply SE(3) transformation to points.
    
    Args:
        q: (..., 4) quaternion
        t: (..., 3) translation
        points: (..., N, 3) points
    
    Returns:
        transformed: (..., N, 3) transformed points
    """
    R = quaternion_to_rotation_matrix(q)
    return jnp.einsum('...ij,...nj->...ni', R, points) + t[..., None, :]


# ============================================================================
# Example usage and verification
# ============================================================================

def plot_backbone_frames(
    result: BackboneFrames,
    axis_length: float = 1.5,
    backbone_color: str = 'gray',
    show_residue_labels: bool = True,
    figsize: tuple = (12, 10),
    elev: float = 20,
    azim: float = 45,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot SE(3) frames along the protein backbone in 3D.
    
    Each frame is visualized as three orthogonal arrows:
    - Red (x-axis): Cα → C direction
    - Green (y-axis): perpendicular in backbone plane
    - Blue (z-axis): perpendicular to backbone plane
    
    Args:
        result: BackboneFrames from pdb_to_se3_frames()
        axis_length: Length of frame axis arrows in Angstroms
        backbone_color: Color for backbone trace
        show_residue_labels: Whether to label residue numbers
        figsize: Figure size
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract valid frames
    valid_idx = np.where(np.array(result.mask))[0]
    translations = np.array(result.translations)
    rotations = np.array(result.rotation_matrices)
    
    # Plot backbone trace (Cα positions)
    valid_trans = translations[valid_idx]
    ax.plot(
        valid_trans[:, 0], 
        valid_trans[:, 1], 
        valid_trans[:, 2],
        color=backbone_color, 
        linewidth=2, 
        alpha=0.6,
        label='Backbone (Cα trace)'
    )
    
    # Plot Cα positions as spheres
    ax.scatter(
        valid_trans[:, 0],
        valid_trans[:, 1],
        valid_trans[:, 2],
        c='black',
        s=30,
        alpha=0.8,
        label='Cα atoms'
    )
    
    # Color scheme for axes
    axis_colors = ['#e41a1c', '#4daf4a', '#377eb8']  # Red, Green, Blue
    axis_labels = ['x (Cα→C)', 'y (in-plane)', 'z (normal)']
    
    # Plot coordinate frames
    for i, idx in enumerate(valid_idx):
        origin = translations[idx]
        R = rotations[idx]
        
        # Draw each axis as an arrow
        for j in range(3):
            axis_dir = R[:, j] * axis_length
            ax.quiver(
                origin[0], origin[1], origin[2],
                axis_dir[0], axis_dir[1], axis_dir[2],
                color=axis_colors[j],
                arrow_length_ratio=0.15,
                linewidth=1.5,
                alpha=0.8,
            )
        
        # Add residue label
        if show_residue_labels:
            ax.text(
                origin[0], origin[1], origin[2] + axis_length * 0.3,
                str(idx + 1),
                fontsize=8,
                ha='center',
                alpha=0.7
            )
    
    # Create legend entries for axes
    for j in range(3):
        ax.quiver([], [], [], [], [], [], color=axis_colors[j], label=axis_labels[j])
    
    # Set labels and title
    ax.set_xlabel('X (Å)', fontsize=10)
    ax.set_ylabel('Y (Å)', fontsize=10)
    ax.set_zlabel('Z (Å)', fontsize=10)
    ax.set_title('Protein Backbone SE(3) Frames\n(Quaternion + Translation representation)', fontsize=12)
    
    # Set equal aspect ratio
    max_range = np.array([
        valid_trans[:, 0].max() - valid_trans[:, 0].min(),
        valid_trans[:, 1].max() - valid_trans[:, 1].min(),
        valid_trans[:, 2].max() - valid_trans[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (valid_trans[:, 0].max() + valid_trans[:, 0].min()) * 0.5
    mid_y = (valid_trans[:, 1].max() + valid_trans[:, 1].min()) * 0.5
    mid_z = (valid_trans[:, 2].max() + valid_trans[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range - axis_length, mid_x + max_range + axis_length)
    ax.set_ylim(mid_y - max_range - axis_length, mid_y + max_range + axis_length)
    ax.set_zlim(mid_z - max_range - axis_length, mid_z + max_range + axis_length)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # if save_path:
        # plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # print(f"Saved figure to {save_path}")
    
    plt.show()

def trajectory_to_se3_frames(
    topology_path: str,
    trajectory_path: str,
    stride: int = 1,
) -> list[jnp.ndarray]:
    """
    Convert an MD trajectory to a list of SE(3) frames.
    
    Uses MDTraj to load the trajectory. Supports common formats:
    - Topology: PDB, GRO, PSF, PRMTOP, etc.
    - Trajectory: XTC, TRR, DCD, NC, HDF5, etc.
    
    Args:
        topology_path: Path to topology file (e.g., PDB, GRO)
        trajectory_path: Path to trajectory file (e.g., XTC, DCD)
                        Can be same as topology_path for multi-model PDB
        stride: Load every nth frame (default: 1 = all frames)
    
    Returns:
        List of K arrays, each of shape (N_res, 7) with [w, x, y, z, tx, ty, tz]
    """
    
    # Load trajectory
    if topology_path == trajectory_path:
        # Single file (e.g., multi-model PDB)
        traj = md.load(trajectory_path, stride=stride)
    else:
        traj = md.load(trajectory_path, top=topology_path, stride=stride)
    
    # Get atom indices for backbone atoms
    # MDTraj uses nanometers, we'll convert to Angstroms
    topology = traj.topology
    
    # Build residue -> backbone atom index mapping
    n_residues = topology.n_residues
    n_indices = np.full(n_residues, -1, dtype=int)
    ca_indices = np.full(n_residues, -1, dtype=int)
    c_indices = np.full(n_residues, -1, dtype=int)
    
    for residue in topology.residues:
        res_idx = residue.index
        for atom in residue.atoms:
            if atom.name == 'N':
                n_indices[res_idx] = atom.index
            elif atom.name == 'CA':
                ca_indices[res_idx] = atom.index
            elif atom.name == 'C':
                c_indices[res_idx] = atom.index
    
    # Process each frame
    frames_list = []
    
    for frame_idx in range(traj.n_frames):
        # Get coordinates for this frame (convert nm -> Å)
        coords = traj.xyz[frame_idx] * 10.0  # nm to Angstroms
        
        # Extract backbone coordinates
        n_coords = np.full((n_residues, 3), np.nan)
        ca_coords = np.full((n_residues, 3), np.nan)
        c_coords = np.full((n_residues, 3), np.nan)
        
        for i in range(n_residues):
            if n_indices[i] >= 0:
                n_coords[i] = coords[n_indices[i]]
            if ca_indices[i] >= 0:
                ca_coords[i] = coords[ca_indices[i]]
            if c_indices[i] >= 0:
                c_coords[i] = coords[c_indices[i]]
        
        # Convert to JAX arrays
        n_coords_jax = jnp.array(n_coords)
        ca_coords_jax = jnp.array(ca_coords)
        c_coords_jax = jnp.array(c_coords)
        
        # Construct frames
        rotation_matrices, translations, mask = construct_backbone_frames(
            n_coords_jax, ca_coords_jax, c_coords_jax
        )
        
        # Convert to quaternions
        quaternions = rotation_matrix_to_quaternion(rotation_matrices)
        
        # Handle invalid residues
        identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        quaternions = jnp.where(mask[:, None], quaternions, identity_quat)
        
        # Concatenate to (N_res, 7)
        frames = jnp.concatenate([quaternions, translations], axis=-1)
        frames_list.append(frames)
    
    return frames_list


def trajectory_to_se3_frames_stacked(
    topology_path: str,
    trajectory_path: str,
    stride: int = 1,
) -> jnp.ndarray:
    """
    Convert an MD trajectory to a stacked array of SE(3) frames.
    
    Same as trajectory_to_se3_frames but returns a single (K, N_res, 7) array
    instead of a list, which is more convenient for batch operations.
    
    Args:
        topology_path: Path to topology file
        trajectory_path: Path to trajectory file
        stride: Load every nth frame
    
    Returns:
        Array of shape (K, N_res, 7) with [w, x, y, z, tx, ty, tz] per residue
    """
    frames_list = trajectory_to_se3_frames(topology_path, trajectory_path, stride)
    return jnp.stack(frames_list, axis=0)    

# ============================================================================
# Example usage and verification
# ============================================================================

if __name__ == "__main__":
    import sys
    pdb_path = sys.argv[1]
    
    # Convert to SE(3) frames
    print(f"Processing: {pdb_path}")
    print("=" * 60)
    
    result = pdb_to_se3_frames(pdb_path)
    
    print(f"\nNumber of residues: {len(result.mask)}")
    print(f"Valid residues: {result.mask.sum()}")
    print(f"\nFrames shape: {result.frames.shape}")
    print(f"  - Quaternions: {result.quaternions.shape}")
    print(f"  - Translations: {result.translations.shape}")
    
    print("\n" + "=" * 60)
    print("Per-residue SE(3) frames (quaternion [w,x,y,z], translation [x,y,z]):")
    print("=" * 60)
    
    for i in range(len(result.mask)):
        if result.mask[i]:
            q = result.quaternions[i]
            t = result.translations[i]
            print(f"\nResidue {i+1}:")
            print(f"  Quaternion: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
            print(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
            print(f"  Combined (7D): {result.frames[i]}")
    
    # Verification: check quaternion properties
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)
    
    # Check unit quaternions
    quat_norms = jnp.linalg.norm(result.quaternions, axis=-1)
    print(f"\nQuaternion norms (should be 1.0): {quat_norms}")
    
    # Check rotation matrix properties
    for i in range(len(result.mask)):
        if result.mask[i]:
            R = result.rotation_matrices[i]
            det = jnp.linalg.det(R)
            orthogonality = jnp.linalg.norm(R @ R.T - jnp.eye(3))
            print(f"Residue {i+1}: det(R) = {det:.6f}, ||RR^T - I|| = {orthogonality:.6f}")
    
    # Verify quaternion ↔ rotation matrix roundtrip
    print("\nRoundtrip verification (quat → matrix → quat):")
    R_reconstructed = quaternion_to_rotation_matrix(result.quaternions)
    q_roundtrip = rotation_matrix_to_quaternion(R_reconstructed)
    
    for i in range(len(result.mask)):
        if result.mask[i]:
            diff = jnp.abs(result.quaternions[i]) - jnp.abs(q_roundtrip[i])
            print(f"Residue {i+1}: max |Δq| = {jnp.abs(diff).max():.2e}")
    
    # Plot the frames
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # 3D plot
    plot_backbone_frames(
        result,
        axis_length=1.5,
        save_path="/tmp/backbone_frames_3d.png"
    )
    
    # 2D projections
    # plot_frames_2d_projections(
    #     result,
    #     axis_length=1.5,
    #     save_path="/tmp/backbone_frames_2d.png"
    # )