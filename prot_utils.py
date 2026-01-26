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
        print (len(atom_names), len(atomtypes))
        ca_indices = np.where(np.array(atomtypes) == 'CA')[0]
        n_indices = np.where(np.array(atomtypes) == 'N')[0]
        c_indices = np.where(np.array(atomtypes) == 'C')[0]
        
        # Get coordinates: shape (n_frames, n_atoms, 3)
        coords = domain[temperature][replica]['coords'][:]
        
        # Extract only CA atoms
        ca_coords = coords[:, ca_indices, :]
        n_coords = coords[:, n_indices, :]
        c_coords = coords[:, c_indices, :]

        # get amino acid residue sequence as a string
        
        res_mapper = {
            'ALA': 'A',  # Alanine
            'ARG': 'R',  # Arginine
            'ASN': 'N',  # Asparagine
            'ASP': 'D',  # Aspartic acid
            'CYS': 'C',  # Cysteine
            'GLN': 'Q',  # Glutamine
            'GLU': 'E',  # Glutamic acid
            'GLY': 'G',  # Glycine
            'HIS': 'H',  # Histidine
            'ILE': 'I',  # Isoleucine
            'LEU': 'L',  # Leucine
            'LYS': 'K',  # Lysine
            'MET': 'M',  # Methionine
            'PHE': 'F',  # Phenylalanine
            'PRO': 'P',  # Proline
            'SER': 'S',  # Serine
            'THR': 'T',  # Threonine
            'TRP': 'W',  # Tryptophan
            'TYR': 'Y',  # Tyrosine
            'VAL': 'V',  # Valine
            # Protonation states (common in MD simulations)
            'HSD': 'H',  # Histidine (delta protonated)
            'HSE': 'H',  # Histidine (epsilon protonated)
            'HSP': 'H',  # Histidine (doubly protonated)
            'HID': 'H',  # Alternative naming
            'HIE': 'H',
            'HIP': 'H',
        }
        aa_mapper = lambda res_list: list(map(lambda res : res_mapper[res], res_list))
        sequence = domain['resname'][()].astype(str)[ca_indices].tolist()
        sequence = "".join(aa_mapper(sequence))
        
    return ca_coords, n_coords, c_coords, domain_id, sequence

import jax.numpy as jnp
import numpy as np

def quat_to_rotmat(q):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    return jnp.stack([
        jnp.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], axis=-1),
        jnp.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], axis=-1),
        jnp.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], axis=-1),
    ], axis=-2)

def local_backbone():
    """Idealized N, CA, C in local frame (CA at origin)."""
    return jnp.array([
        [-1.46, 0.0, 0.0],
        [0.0,   0.0, 0.0],
        [1.52,  0.0, 0.0],
    ])

def trajectory_to_coords(trajectory):
    """
    Convert SE(3) trajectory to backbone coordinates.
    
    Args:
        trajectory: (T, N, 7) - [quat_w, quat_x, quat_y, quat_z, tx, ty, tz]
    
    Returns:
        coords: (T, N, 3, 3) - N, CA, C for each residue at each timestep
    """
    quats = trajectory[..., :4]        # (T, N, 4)
    trans = trajectory[..., 4:]        # (T, N, 3)
    
    rotmats = quat_to_rotmat(quats)    # (T, N, 3, 3)
    local_atoms = local_backbone()     # (3, 3)
    
    # Apply frames: (T, N, 3, 3) @ (3, 3).T + (T, N, 1, 3)
    coords = jnp.einsum('tnij,kj->tnki', rotmats, local_atoms) + trans[:, :, None, :]
    return coords

def save_trajectory_pdb(coords, filename):
    """
    Save trajectory to multi-model PDB.
    
    Args:
        coords: (T, N, 3, 3) - trajectory of backbone coordinates
        filename: output path
    """
    coords = np.array(coords)
    T, N, _, _ = coords.shape
    atom_names = ['N', 'CA', 'C']
    
    with open(filename, 'w') as f:
        for t in range(T):
            f.write(f"MODEL     {t+1:4d}\n")
            atom_idx = 1
            
            for res_idx in range(N):
                for atom_name, xyz in zip(atom_names, coords[t, res_idx]):
                    f.write(
                        f"ATOM  {atom_idx:5d}  {atom_name:<3s} ALA A{res_idx+1:4d}    "
                        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                        f"  1.00  0.00           {atom_name[0]:>2s}\n"
                    )
                    atom_idx += 1
            
            f.write("ENDMDL\n")
        f.write("END\n")

def tic_plot():
    pass