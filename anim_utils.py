"""
3D Protein MD Trajectory Animation
Visualizes Cα backbone as a connected chain that jiggles through time.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def animate_protein_trajectory(
    trajectory: np.ndarray,
    interval: int = 50,
    figsize: tuple = (10, 8),
    color: str = "royalblue",
    line_width: float = 2.0,
    atom_size: float = 30,
    save_path: str = None,
    fps: int = 20,
    elev: float = 20,
    azim: float = 45,
    rotate: bool = True,
    rotation_speed: float = 0.5,
):
    """
    Animate a 3D protein MD trajectory.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Shape (T, N_atoms, 3) - the MD trajectory
    interval : int
        Milliseconds between frames
    figsize : tuple
        Figure size
    color : str
        Color for the backbone
    line_width : float
        Width of backbone lines
    atom_size : float
        Size of Cα atom markers
    save_path : str
        If provided, save animation to this path (e.g., 'protein.gif')
    fps : int
        Frames per second for saved animation
    elev : float
        Initial elevation angle
    azim : float
        Initial azimuth angle
    rotate : bool
        Whether to rotate the view during animation
    rotation_speed : float
        Speed of rotation in degrees per frame
    
    Returns
    -------
    FuncAnimation object
    """
    T, N_atoms, _ = trajectory.shape
    
    # Center the trajectory
    centroid = trajectory.mean(axis=(0, 1), keepdims=True)
    trajectory_centered = trajectory - centroid
    
    # Compute global axis limits for consistent view
    all_coords = trajectory_centered.reshape(-1, 3)
    max_range = np.abs(all_coords).max() * 1.1
    
    # Set up figure
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Style the axes
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3)
    
    # Set consistent limits
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_xlabel('X (Å)', fontsize=10)
    ax.set_ylabel('Y (Å)', fontsize=10)
    ax.set_zlabel('Z (Å)', fontsize=10)
    
    # Initial view
    ax.view_init(elev=elev, azim=azim)
    
    # Initialize plot elements
    coords = trajectory_centered[0]
    
    # Backbone line (connected Cα atoms)
    line, = ax.plot(
        coords[:, 0], coords[:, 1], coords[:, 2],
        color=color, linewidth=line_width, alpha=0.8
    )
    
    # Cα atoms as scatter points
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=np.arange(N_atoms), cmap='viridis', s=atom_size,
        edgecolors='white', linewidths=0.5, alpha=0.9
    )
    
    # Title
    title = ax.set_title(f'Frame 0/{T-1}', fontsize=12, fontweight='bold')
    
    def update(frame):
        coords = trajectory_centered[frame]
        
        # Update backbone line
        line.set_data(coords[:, 0], coords[:, 1])
        line.set_3d_properties(coords[:, 2])
        
        # Update scatter points
        scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        
        # Update title
        title.set_text(f'Frame {frame}/{T-1}')
        
        # Rotate view if enabled
        if rotate:
            ax.view_init(elev=elev, azim=azim + frame * rotation_speed)
        
        return line, scatter, title
    
    anim = FuncAnimation(
        fig, update, frames=T,
        interval=interval, blit=False
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)
    
    plt.tight_layout()
    return anim, fig


def generate_example_trajectory(n_frames=100, n_atoms=50, amplitude=0.5):
    """
    Generate a synthetic protein trajectory for demonstration.
    Creates a helical protein that wiggles over time.
    """
    # Create initial helical structure
    t = np.linspace(0, 4 * np.pi, n_atoms)
    x = 5 * np.cos(t)
    y = 5 * np.sin(t)
    z = np.linspace(0, 20, n_atoms)
    
    initial_coords = np.stack([x, y, z], axis=-1)
    
    # Generate trajectory with correlated noise (jiggly motion)
    trajectory = np.zeros((n_frames, n_atoms, 3))
    
    # Use multiple frequency components for realistic motion
    for i in range(n_frames):
        phase = i * 2 * np.pi / n_frames
        
        # Low frequency global motion
        global_motion = amplitude * np.sin(phase + t[:, None] * 0.5) * np.array([1, 1, 0.3])
        
        # Higher frequency local wiggling
        local_wiggle = 0.3 * amplitude * np.sin(3 * phase + t[:, None] * 2) * np.array([1, 1, 0.5])
        
        # Random thermal noise (small)
        noise = 0.1 * amplitude * np.random.randn(n_atoms, 3)
        
        trajectory[i] = initial_coords + global_motion + local_wiggle + noise
    
    return trajectory


# Main execution
if __name__ == "__main__":
    # Generate example trajectory
    print("Generating example protein trajectory...")
    trajectory = generate_example_trajectory(n_frames=100, n_atoms=60, amplitude=1.0)
    print (trajectory.shape)
    print(f"Trajectory shape: {trajectory.shape}")
    
    # Create animation and save as GIF
    anim, fig = animate_protein_trajectory(
        trajectory,
        interval=50,
        color='royalblue',
        line_width=2.5,
        atom_size=40,
        save_path='protein_trajectory.gif',
        fps=20,
        rotate=True,
        rotation_speed=1.0
    )
    
    print("Animation saved!")