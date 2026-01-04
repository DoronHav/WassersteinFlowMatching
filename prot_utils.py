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
import h5py

def atom2frame(n_coords, ca_coords, c_coords):
    """
    Convert backbone atom coordinates to SE(3) frames per residue.
    Follows FrameDiff/AlphaFold2 rigidFrom3Point algorithm.
    
    Args:
        n_coords:  (T, N_residues, 3) N atom positions
        ca_coords: (T, N_residues, 3) CA atom positions
        c_coords:  (T, N_residues, 3) C atom positions
    
    Returns:
        se3_frames: (T, N_residues, 7) array where each frame is
                    [qw, qx, qy, qz, tx, ty, tz]
    """
    eps = 1e-8
    
    # v1 = C - CA, v2 = N - CA
    v1 = c_coords - ca_coords
    v2 = n_coords - ca_coords
    
    # e1 = v1 / ||v1||  (note: paper has typo, normalizes by ||v2||, but should be ||v1||)
    e1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + eps)
    
    # u2 = v2 - e1 * (e1^T @ v2)  -- subtract projection onto e1
    dot = np.sum(e1 * v2, axis=-1, keepdims=True)  # e1^T @ v2
    u2 = v2 - e1 * dot
    
    # e2 = u2 / ||u2||
    e2 = u2 / (np.linalg.norm(u2, axis=-1, keepdims=True) + eps)
    
    # e3 = e1 x e2
    e3 = np.cross(e1, e2)
    
    # R = concat(e1, e2, e3) as columns -> (T, N_res, 3, 3)
    rot_matrices = np.stack([e1, e2, e3], axis=-1)
    
    # Translation = CA position
    translation = ca_coords
    
    # Convert rotation matrices to quaternions
    quaternions = rotation_matrix_to_quaternion(rot_matrices)
    
    # Concatenate [quat, translation] -> (T, N_res, 7)
    se3_frames = np.concatenate([quaternions, translation], axis=-1)
    
    return se3_frames

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrices to quaternions (wxyz convention).
    
    Args:
        R: (..., 3, 3) rotation matrices
    
    Returns:
        q: (..., 4) quaternions [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    N = R.shape[0]
    
    q = np.zeros((N, 4), dtype=R.dtype)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = np.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1
    
    # Case 2: R[0,0] largest diagonal
    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = np.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2
    
    # Case 3: R[1,1] largest diagonal
    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    s3 = np.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3
    
    # Case 4: R[2,2] largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    s4 = np.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4
    
    # Normalize and ensure w >= 0
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    q[q[:, 0] < 0] *= -1
    
    return q.reshape(*batch_shape, 4)

def get_ca_trajectory(h5_path, temperature='320', replica='0'):
    """
    Extract C-alpha atom trajectory from an mdCATH H5 file.
    
    Args:
        h5_path: Path to the mdCATH H5 file
        temperature: Temperature group (320, 348, 379, 413, 450)
        replica: Replica number (0-4)
    
    Returns:
        ca_coords: numpy array of shape (n_frames, n_ca_atoms, 3)
    """
    with h5py.File(h5_path, 'r') as f:
        # Get the domain ID (root group)
        domain_id = list(f.keys())[0]
        domain = f[domain_id]
        
        # Get atom names to find CA indices
        # 'element' contains atom element symbols, but we need atom names
        # In mdCATH, atom names are typically in the pdbProteinAtoms or similar
        # The 'name' field contains atom names like 'CA', 'N', 'C', 'O', etc.
        atom_names = domain['pdbProteinAtoms'][()].decode('utf-8').split('\n')[1:-3] # remove header and footer
        atomtypes = [line.split()[2] for line in atom_names]
        ca_indices = np.where(np.array(atomtypes) == 'CA')[0]
        n_indices = np.where(np.array(atomtypes) == 'N')[0]
        c_indices = np.where(np.array(atomtypes) == 'C')[0]
        
        # Get coordinates: shape (n_frames, n_atoms, 3)
        coords = domain[temperature][replica]['coords'][:]
        
        # Extract only CA atoms
        ca_coords = coords[:, ca_indices, :]
        n_coords = coords[:, n_indices, :]
        c_coords = coords[:, c_indices, :]
        
    return ca_coords, n_coords, c_coords, domain_id