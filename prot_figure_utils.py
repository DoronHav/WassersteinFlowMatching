import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, FastICA
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

# -----------------------------
# 1. Dihedral computation
# -----------------------------
def compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle between 4 points. Returns radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)
    
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8))
    
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    
    return np.arctan2(y, x)


def extract_phi_psi(N, CA, C):
    """
    N, CA, C: (n_frames, n_residues, 3)
    Returns phi, psi in radians
    """
    phi = compute_dihedral(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])
    psi = compute_dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])
    return phi, psi


def torsions_to_features(phi, psi):
    """Cos/sin encoding for dimensionality reduction."""
    phi_overlap = phi[:, :-1]
    psi_overlap = psi[:, 1:]
    return np.concatenate([
        np.cos(phi_overlap), np.sin(phi_overlap),
        np.cos(psi_overlap), np.sin(psi_overlap)
    ], axis=1)


# -----------------------------
# 2. TICA implementation
# -----------------------------
class SimpleTICA:
    """Minimal TICA for MD trajectories."""
    def __init__(self, lag=1, dim=2):
        self.lag = lag
        self.dim = dim
        
    def fit(self, X):
        n = X.shape[0]
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        C0 = X_centered.T @ X_centered / (n - 1)
        
        X_t = X_centered[:-self.lag]
        X_tau = X_centered[self.lag:]
        C_tau = X_t.T @ X_tau / (n - self.lag - 1)
        C_tau = (C_tau + C_tau.T) / 2
        
        C0_reg = C0 + 1e-6 * np.eye(C0.shape[0])
        eigvals_c0, eigvecs_c0 = np.linalg.eigh(C0_reg)
        eigvals_c0 = np.maximum(eigvals_c0, 1e-10)
        whitening = eigvecs_c0 @ np.diag(1.0 / np.sqrt(eigvals_c0)) @ eigvecs_c0.T
        
        C_tau_white = whitening @ C_tau @ whitening.T
        eigvals, eigvecs = np.linalg.eigh(C_tau_white)
        
        idx = np.argsort(eigvals)[::-1]
        self.eigenvectors_ = whitening.T @ eigvecs[:, idx][:, :self.dim]
        return self
    
    def transform(self, X):
        return (X - self.mean_) @ self.eigenvectors_

def plot_ramachandran(ax, phi, psi, title='', color='blue', alpha=0.3):
    """
    Ramachandran plot for ensemble.
    
    phi, psi: (n_frames, n_residues) in radians
    """
    # Flatten all residues across all frames
    phi_flat = np.degrees(phi.flatten())
    psi_flat = np.degrees(psi.flatten())
    
    # Remove any NaN
    valid = ~(np.isnan(phi_flat) | np.isnan(psi_flat))
    phi_flat = phi_flat[valid]
    psi_flat = psi_flat[valid]
    
    # 2D histogram as density
    H, xedges, yedges = np.histogram2d(
        phi_flat, psi_flat, 
        bins=72,  # 5-degree bins
        range=[[-180, 180], [-180, 180]],
        density=True
    )
    
    # Plot as filled contour
    X, Y = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2
    )
    
    ax.contourf(X, Y, H.T, levels=20, cmap='Blues' if color == 'blue' else 'Oranges', alpha=0.8)
    
    # Standard Ramachandran regions (approximate)
    # Alpha helix region
    ax.add_patch(plt.Rectangle((-80, -60), 40, 40, fill=False, edgecolor='gray', linestyle='--', linewidth=0.8))
    # Beta sheet region  
    ax.add_patch(plt.Rectangle((-150, 100), 80, 60, fill=False, edgecolor='gray', linestyle='--', linewidth=0.8))
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel('φ (degrees)')
    ax.set_ylabel('ψ (degrees)')
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linewidth=0.3)
    ax.axvline(0, color='k', linewidth=0.3)


def plot_ramachandran_comparison(ref_N, ref_CA, ref_C, gen_N, gen_CA, gen_C, protein_name=''):
    """Side-by-side Ramachandran comparison."""
    
    ref_phi, ref_psi = extract_phi_psi(ref_N, ref_CA, ref_C)
    gen_phi, gen_psi = extract_phi_psi(gen_N, gen_CA, gen_C)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    plot_ramachandran(axes[0], ref_phi, ref_psi, title='Reference MD', color='blue')
    plot_ramachandran(axes[1], gen_phi, gen_psi, title='Generated', color='orange')
    
    fig.suptitle(f'{protein_name} Ramachandran', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes


def plot_ramachandran_overlay(ref_N, ref_CA, ref_C, gen_N, gen_CA, gen_C, protein_name=''):
    """
    Overlay Ramachandran with contour lines for comparison.
    This is often more useful than side-by-side.
    """
    ref_phi, ref_psi = extract_phi_psi(ref_N, ref_CA, ref_C)
    gen_phi, gen_psi = extract_phi_psi(gen_N, gen_CA, gen_C)
    
    ref_phi_flat = np.degrees(ref_phi.flatten())
    ref_psi_flat = np.degrees(ref_psi.flatten())
    gen_phi_flat = np.degrees(gen_phi.flatten())
    gen_psi_flat = np.degrees(gen_psi.flatten())
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    # Reference as filled contour
    H_ref, xedges, yedges = np.histogram2d(
        ref_phi_flat, ref_psi_flat,
        bins=72, range=[[-180, 180], [-180, 180]], density=True
    )
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
    ax.contourf(X, Y, H_ref.T, levels=15, cmap='Blues', alpha=0.7)
    
    # Generated as contour lines
    H_gen, _, _ = np.histogram2d(
        gen_phi_flat, gen_psi_flat,
        bins=72, range=[[-180, 180], [-180, 180]], density=True
    )
    ax.contour(X, Y, H_gen.T, levels=8, colors='orangered', linewidths=1.2)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel('φ (degrees)', fontsize=11)
    ax.set_ylabel('ψ (degrees)', fontsize=11)
    ax.set_title(f'{protein_name}', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linewidth=0.3)
    ax.axvline(0, color='k', linewidth=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=8, alpha=0.5, label='MD'),
        Line2D([0], [0], color='orangered', lw=2, label='Generated')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    plt.tight_layout()
    return fig, ax

def create_fes_plot(ref_N, ref_CA, ref_C, gen_N, gen_CA, gen_C, protein_name='', bins=100, sigma=2.0):
    # Coordinates → angles → features
    ref_phi, ref_psi = extract_phi_psi(ref_N, ref_CA, ref_C)
    gen_phi, gen_psi = extract_phi_psi(gen_N, gen_CA, gen_C)
    
    ref_features = torsions_to_features(ref_phi, ref_psi)
    gen_features = torsions_to_features(gen_phi, gen_psi)
    
    # PCA (fit on reference)
    pca = PCA(n_components=2)
    pca.fit(ref_features)
    
    ref_pca = pca.transform(ref_features)
    gen_pca = pca.transform(gen_features)
    
    # Axis limits
    all_coords = np.vstack([ref_pca, gen_pca])
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    xlim = [x_min - x_pad, x_max + x_pad]
    ylim = [y_min - y_pad, y_max + y_pad]
    
    # Custom colormap
    colors = ['#000033', '#0000FF', '#00BFFF', '#00FF7F', 
              '#ADFF2F', '#FFFF00', '#FFA500', '#FF4500']
    cmap = LinearSegmentedColormap.from_list('fes', colors, N=256)
    
    # Plot - wider figure to accommodate colorbar
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    
    max_energy = 8
    
    for ax, coords, title in [(axes[0], ref_pca, 'Reference MD'), 
                               (axes[1], gen_pca, 'Generated')]:
        
        H, xedges, yedges = np.histogram2d(
            coords[:, 0], coords[:, 1],
            bins=bins,
            range=[xlim, ylim],
            density=True
        )
        
        H_smooth = gaussian_filter(H, sigma=sigma)
        
        kT = 2.494
        H_smooth = np.where(H_smooth > 1e-9, H_smooth, np.nan)
        F = -kT * np.log(H_smooth)
        F = F - np.nanmin(F)
        F = np.clip(F, 0, max_energy)
        F_plot = np.where(np.isnan(F), max_energy, F).T
        
        im = ax.imshow(
            F_plot, 
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
            origin='lower',
            aspect='equal',
            cmap=cmap,
            vmin=0, 
            vmax=max_energy,
            interpolation='gaussian'
        )
        
        ax.set_xlabel('PC 1', fontsize=11)
        ax.set_ylabel('PC 2', fontsize=11)
        ax.set_title(title, fontsize=11, pad=8)
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Colorbar - position it outside the plots
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Free Energy (kJ/mol)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    fig.suptitle(protein_name, fontsize=12, fontweight='bold')
    
    return fig, axes

def create_ica_contour_plot(ref_N, ref_CA, ref_C, gen_N, gen_CA, gen_C,
                            protein_name='', n_components=2):
    """
    ICA-based contour plot like FoldFlow Figure 5b.
    """
    # Coordinates → angles → features
    ref_phi, ref_psi = extract_phi_psi(ref_N, ref_CA, ref_C)
    gen_phi, gen_psi = extract_phi_psi(gen_N, gen_CA, gen_C)
    
    ref_features = torsions_to_features(ref_phi, ref_psi)
    gen_features = torsions_to_features(gen_phi, gen_psi)
    
    # ICA (fit on reference)
    ica = FastICA(n_components=n_components, random_state=42)
    ica.fit(ref_features)
    
    ref_ica = ica.transform(ref_features)
    gen_ica = ica.transform(gen_features)
    
    # Axis limits
    all_coords = np.vstack([ref_ica, gen_ica])
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15
    
    xlim = [x_min - x_pad, x_max + x_pad]
    ylim = [y_min - y_pad, y_max + y_pad]
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    # Grid for KDE
    xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Reference MD - black/gray filled contours
    kde_ref = gaussian_kde(ref_ica.T, bw_method=0.3)
    zz_ref = kde_ref(positions).reshape(xx.shape)
    ax.contour(xx, yy, zz_ref, levels=6, colors='black', linewidths=1.5)
    ax.contourf(xx, yy, zz_ref, levels=6, cmap='Greys', alpha=0.3)
    
    # Generated - red contours
    kde_gen = gaussian_kde(gen_ica.T, bw_method=0.3)
    zz_gen = kde_gen(positions).reshape(xx.shape)
    ax.contour(xx, yy, zz_gen, levels=6, colors='red', linewidths=1.5)
    
    ax.set_xlabel('IC1', fontsize=12)
    ax.set_ylabel('IC2', fontsize=12)
    ax.set_title(protein_name, fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='MD Simulation'),
        Line2D([0], [0], color='red', lw=2, label='Generated'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)
    
    # plt.tight_layout()
    
    return fig, ax

def compute_rmsf(CA):
    """
    Compute per-residue RMSF from CA coordinates.
    
    CA: (n_frames, n_residues, 3)
    Returns: (n_residues,) RMSF values
    """
    # Mean structure
    mean_structure = CA.mean(axis=0)  # (n_residues, 3)
    
    # Deviations from mean
    deviations = CA - mean_structure  # (n_frames, n_residues, 3)
    
    # RMSF = sqrt(mean of squared deviations)
    rmsf = np.sqrt((deviations ** 2).sum(axis=2).mean(axis=0))  # (n_residues,)
    
    return rmsf


def plot_rmsf_comparison(ref_CA, gen_CA, protein_name=''):
    """
    Plot per-residue RMSF comparison with rainbow coloring.
    
    ref_CA: (n_frames, n_residues, 3) - reference ensemble
    gen_CA: (n_frames, n_residues, 3) - generated ensemble
    """
    ref_rmsf = compute_rmsf(ref_CA)
    gen_rmsf = compute_rmsf(gen_CA)
    
    n_residues = len(ref_rmsf)
    residue_idx = np.arange(n_residues)
    
    # Pearson correlation
    r, _ = pearsonr(ref_rmsf, gen_rmsf)
    
    # Rainbow colors by residue index
    colors = plt.cm.rainbow(np.linspace(0, 1, n_residues))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)
    
    for ax, rmsf, title in [(axes[0], ref_rmsf, 'MD'),
                             (axes[1], gen_rmsf, 'Ours')]:
        ax.scatter(residue_idx, rmsf, c=colors, s=30, edgecolors='none')
        ax.set_xlabel('Residue Index', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-1, n_residues)
    
    axes[0].set_ylabel('RMSF (Å)', fontsize=10)
    
    fig.suptitle(f'{protein_name}  (Pearson r = {r:.2f})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes, r