#!/usr/bin/env python3
"""
Tutorial on Riemannian Wasserstein Flow Matching

This script demonstrates how to run RWFM for point-cloud generation on spherical manifolds
using the spherical MNIST dataset.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import matplotlib.pyplot as plt
import skimage
from tqdm import tqdm
from scipy.spatial.distance import cdist
import multiprocessing
from itertools import product

import ott
from ott.solvers import linear
import ot

from wassersteinflowmatching.riemannian_wasserstein import RiemannianWassersteinFlowMatching


def square_to_sphere(points):
    """
    Convert points from a square [-1,1]^2 to spherical coordinates (theta, phi) on a unit sphere.
    
    Args:
        points (np.array): Array of shape (N, 2) containing N points in [-1,1]^2
    
    Returns:
        np.array: Array of shape (N, 3) containing N points as (x, y, z) on the unit sphere
    """
    points = np.array(points)
    
    if np.any(points < -1) or np.any(points > 1):
        points = np.clip(points, -1, 1)
    
    x, y = points[:, 0], points[:, 1]
    
    # Convert to intermediate spherical coordinates
    theta = np.pi * x
    phi = np.pi * (y - 1) / 2
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack((x, y, z))


def spherical_chamfer_distance(args):
    """
    Computes the spherical Chamfer distance between two point clouds.
    Assumes points are on the unit sphere.
    """
    pc1, pc2 = args
    
    # Ensure inputs are numpy arrays
    pc1 = np.asarray(pc1)
    pc2 = np.asarray(pc2)

    # Pairwise dot products
    dot_products = np.clip(pc1 @ pc2.T, -1.0, 1.0)
    
    # Great-circle distances
    distances = np.arccos(dot_products) ** 2
    
    # Minimum distances
    min_dist_pc1_to_pc2 = np.min(distances, axis=1)
    min_dist_pc2_to_pc1 = np.min(distances, axis=0)
    
    # Chamfer distance
    chamfer_dist = np.sum(min_dist_pc1_to_pc2 ** 2) + np.sum(min_dist_pc2_to_pc1 ** 2)
    
    return chamfer_dist


def visualize_point_clouds_on_sphere(point_clouds, num_plots=16, figsize=(20, 20)):
    """
    Visualize point clouds on a sphere.
    
    Args:
        point_clouds: List of point clouds to visualize
        num_plots: Number of point clouds to plot
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Generate sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v)) * 0.98
    y_sphere = np.outer(np.sin(u), np.sin(v)) * 0.98
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.98
    
    for ind in range(min(num_plots, len(point_clouds))):
        ax = fig.add_subplot(4, 4, 1 + ind, projection='3d')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='grey', alpha=0.3, zorder=1)
        
        points = point_clouds[ind]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='magenta', depthshade=True, zorder=2)
        
        ax.view_init(azim=160, elev=-10)
        ax.axis('off')
        plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


class rwfm_config:
    """Configuration for Riemannian Wasserstein Flow Matching"""
    geom: str = 'sphere'
    monge_map: str = 'wasserstein_eps'
    wasserstein_eps: float = 0.005
    wasserstein_lse: bool = True
    num_sinkhorn_iters: int = -1
    mini_batch_ot_mode: bool = True
    mini_batch_ot_solver: str = 'chamfer'
    minibatch_ot_eps: float = 0.01
    minibatch_ot_lse: bool = True
    noise_type: str = 'ambient_gaussian'
    scaling: str = 'None'
    factor: float = 1.0
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 4
    dropout_rate: float = 0.1
    mlp_hidden_dim: int = 512
    cfg: bool = True
    p_cfg_null: float = 0.1
    w_cfg: float = 2.0
    normalized_condition: bool = False


def load_and_preprocess_mnist(data_path='/data/havivd/mnist', label_filter=3):
    """
    Load MNIST images and convert to point clouds on the sphere.
    
    Args:
        data_path: Path to MNIST data files
        label_filter: Which digit to filter for
        
    Returns:
        pc_train, pc_test: Training and test point clouds
    """
    print("Loading MNIST data...")
    image_train = np.load(f'{data_path}/mnist_train_images.npy')
    label_train = np.load(f'{data_path}/mnist_train_labels.npy')
    image_test = np.load(f'{data_path}/mnist_test_images.npy')
    label_test = np.load(f'{data_path}/mnist_test_labels.npy')
    
    # Filter by label
    print(f"Filtering for label {label_filter}...")
    image_train = image_train[label_train == label_filter]
    image_test = image_test[label_test == label_filter]
    
    # Convert to point clouds
    print("Converting images to point clouds...")
    pc_train = [2 * (np.stack(np.where(im > skimage.filters.threshold_otsu(im)), axis=-1)) / 28 - 1 
                for im in tqdm(image_train, desc="Train images")]
    pc_test = [2 * (np.stack(np.where(im > skimage.filters.threshold_otsu(im)), axis=-1)) / 28 - 1 
               for im in tqdm(image_test, desc="Test images")]
    
    # Add noise
    print("Adding noise...")
    pc_train = [np.clip(pc + np.random.normal(0, 0.02, pc.shape), -0.98, 0.98) for pc in pc_train]
    pc_test = [np.clip(pc + np.random.normal(0, 0.02, pc.shape), -0.98, 0.98) for pc in pc_test]
    
    # Rotate coordinates
    pc_train = [np.stack([pc[:, 1], -pc[:, 0]], axis=0).T for pc in pc_train]
    pc_test = [np.stack([pc[:, 1], -pc[:, 0]], axis=0).T for pc in pc_test]
    
    # Map to sphere
    print("Mapping to sphere...")
    pc_train = [square_to_sphere(pc) for pc in pc_train]
    pc_test = [square_to_sphere(pc) for pc in pc_test]
    
    return pc_train, pc_test


def test_noise_generation(flow_model, num_samples=16, visualize=True):
    """Test noise generation on the sphere."""
    print("Testing noise generation...")
    key = random.PRNGKey(0)
    
    K = 1000
    n = 150
    d = 3
    
    key, subkey = random.split(key)
    spherical_points = flow_model.noise_func((K, n, d), flow_model.noise_config, subkey)
    
    if visualize:
        print("Visualizing noise samples...")
        visualize_point_clouds_on_sphere(spherical_points[:num_samples])
    
    return spherical_points, key


def test_ot_map(flow_model, key, visualize=True):
    """Test optimal transport map."""
    print("Testing OT map...")
    
    batch_ind = np.random.choice(np.arange(len(flow_model.point_clouds)), size=16, replace=False)
    point_clouds_batch = flow_model.point_clouds[batch_ind]
    weights_batch = flow_model.weights[batch_ind]
    
    subkey_noise, key = random.split(key)
    noise_samples = flow_model.noise_func(
        size=point_clouds_batch.shape,
        noise_config=flow_model.noise_config,
        key=subkey_noise
    )
    noise_weights = weights_batch
    noise_samples = flow_model.project_to_geometry(noise_samples)
    
    subkey_t, key = random.split(key)
    interpolates_time = random.uniform(subkey_t, (point_clouds_batch.shape[0],), minval=0.0, maxval=1.0)
    
    ot_matrix, ot_solve = flow_model.transport_plan_jit(
        [noise_samples, noise_weights],
        [point_clouds_batch, weights_batch]
    )
    
    # Check which assignment method is being used
    if flow_model.monge_map == 'sample':
        assigned_points = flow_model.sample_map_jit(
            ot_matrix, point_clouds_batch, random.split(key, point_clouds_batch.shape[0])
        )
    else:
        assigned_points = flow_model.sample_map_jit(ot_matrix, point_clouds_batch)
    
    point_cloud_interpolates = flow_model.interpolant_vmap(
        noise_samples, assigned_points, 1 - interpolates_time
    )
    point_cloud_velocity = flow_model.interpolant_velocity_vmap(
        noise_samples, assigned_points, 1 - interpolates_time
    )
    
    print(f"OT converged: {ot_solve.converged}")
    
    if visualize:
        print("Visualizing interpolation...")
        ind_plot = np.random.choice(np.arange(point_cloud_interpolates.shape[0]))
        
        point_cloud_end_point = jnp.squeeze(
            flow_model.exponential_map_vmap(
                point_cloud_interpolates[ind_plot][None, :, :],
                point_cloud_velocity[ind_plot][None, :, :],
                interpolates_time[ind_plot]
            )
        )
        
        # Generate sphere wireframe
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v)) * 0.98
        y = np.outer(np.sin(u), np.sin(v)) * 0.98
        z = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.98
        
        fig = plt.figure(figsize=(20, 5))
        
        # Plot noise
        ax = fig.add_subplot(1, 4, 1, projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.plot_wireframe(x, y, z, color='grey', alpha=0.3, zorder=1)
        points = noise_samples[ind_plot][noise_weights[ind_plot] > 0]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='magenta', depthshade=True, zorder=2, s=10)
        ax.view_init(azim=160, elev=-10)
        ax.axis('off')
        plt.title(f'Time: 1.00')
        plt.gca().set_aspect('equal')
        
        # Plot interpolate
        ax = fig.add_subplot(1, 4, 2, projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.plot_wireframe(x, y, z, color='grey', alpha=0.3, zorder=1)
        points = point_cloud_interpolates[ind_plot][noise_weights[ind_plot] > 0]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='magenta', depthshade=True, zorder=2, s=10)
        ax.view_init(azim=160, elev=-10)
        ax.axis('off')
        plt.title(f'Time: {interpolates_time[ind_plot]:.2f}')
        plt.gca().set_aspect('equal')
        
        # Plot predicted endpoint
        ax = fig.add_subplot(1, 4, 3, projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.plot_wireframe(x, y, z, color='grey', alpha=0.3, zorder=1)
        points = point_cloud_end_point[noise_weights[ind_plot] > 0]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='magenta', depthshade=True, zorder=2, s=10)
        ax.view_init(azim=160, elev=-10)
        ax.axis('off')
        plt.title(f'Time: 0.00, predicted')
        plt.gca().set_aspect('equal')
        
        # Plot target
        ax = fig.add_subplot(1, 4, 4, projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.plot_wireframe(x, y, z, color='grey', alpha=0.3, zorder=1)
        points = point_clouds_batch[ind_plot][weights_batch[ind_plot] > 0]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='magenta', depthshade=True, zorder=2, s=10)
        ax.view_init(azim=160, elev=-10)
        ax.axis('off')
        plt.title(f'Time: 0.00')
        plt.gca().set_aspect('equal')
        plt.show()
    
    return key


def train_model(flow_model, batch_size=32, training_steps=200000, decay_steps=2000):
    """Train the flow matching model."""
    print(f"Training model for {training_steps} steps...")
    flow_model.train(
        batch_size=batch_size,
        training_steps=training_steps,
        decay_steps=decay_steps
    )
    
    # Plot loss curve
    loss_smooth = np.convolve(np.log(flow_model.losses), np.ones(100) / 100, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(loss_smooth)
    plt.xlabel('Training Step')
    plt.ylabel('Log Loss (smoothed)')
    plt.title('Training Loss Curve')
    plt.show()


def generate_and_visualize_samples(flow_model, key, num_samples=16, timesteps=1000):
    """Generate samples and visualize them."""
    print(f"Generating {num_samples} samples with {timesteps} timesteps...")
    
    subkey, key = jax.random.split(key)
    generated_samples, sample_weights = flow_model.generate_samples(
        num_samples=num_samples,
        timesteps=timesteps,
        key=subkey
    )
    
    print("Visualizing generated samples...")
    final_samples = [generated_samples[-1][i][sample_weights[i] > 0] for i in range(num_samples)]
    visualize_point_clouds_on_sphere(final_samples)
    
    return generated_samples, sample_weights, key


def benchmark_model(flow_model, pc_test, key, batch_size=128):
    """Benchmark the model using Chamfer distance and nearest neighbor accuracy."""
    print("Benchmarking model...")
    
    num_generate = len(pc_test)
    all_generated_samples = []
    all_sample_weights = []
    
    for batch_ind in range(num_generate // batch_size + 1):
        print(f'Generating batch {batch_ind + 1}/{num_generate // batch_size + 1}')
        subkey, key = jax.random.split(key)
        generated_samples, sample_weights = flow_model.generate_samples(
            num_samples=batch_size,
            timesteps=1000,
            key=subkey
        )
        all_generated_samples.append(generated_samples[-1])
        all_sample_weights.append(sample_weights)
    
    all_generated_samples = jnp.concatenate(all_generated_samples, axis=0)[:num_generate]
    all_sample_weights = jnp.concatenate(all_sample_weights, axis=0)[:num_generate]
    
    # Prepare point clouds for comparison
    print("Preparing point clouds...")
    pc_test_list = [np.asarray(pc) for pc in pc_test]
    generated_samples_list = [
        np.asarray(all_generated_samples[i][all_sample_weights[i] > 0]) 
        for i in range(len(all_generated_samples))
    ]
    
    # Combine both lists
    all_point_clouds = generated_samples_list + pc_test_list
    
    # Create all pairs
    pairs = list(product(all_point_clouds, repeat=2))
    
    # Compute distance matrix
    print("Computing Chamfer distance matrix...")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        distances = list(tqdm(
            pool.imap(spherical_chamfer_distance, pairs),
            total=len(pairs),
            desc="Computing Full CD Matrix"
        ))
    
    # Reshape into matrix
    num_total = len(all_point_clouds)
    cd_distance_matrix = np.array(distances).reshape((num_total, num_total))
    
    print(f"Shape of the full CD distance matrix: {cd_distance_matrix.shape}")
    
    # Compute nearest neighbor accuracy
    label = np.asarray(['generated'] * len(generated_samples_list) + ['real'] * len(pc_test_list))
    nearest_neighbor_label = label[np.argsort(cd_distance_matrix, axis=1)[:, 1]]
    
    nna = np.mean(nearest_neighbor_label == label)
    print(f"Nearest neighbor accuracy: {nna:.4f}")
    
    return nna, cd_distance_matrix, key


def main():
    """Main execution function."""
    print("=" * 80)
    print("Spherical MNIST Riemannian Wasserstein Flow Matching Tutorial")
    print("=" * 80)
    
    # Load and preprocess data
    pc_train, pc_test = load_and_preprocess_mnist()
    print(f"Training samples: {len(pc_train)}, Test samples: {len(pc_test)}")
    
    # Visualize some training samples
    print("\nVisualizing training samples...")
    sample_indices = np.random.choice(len(pc_train), size=16, replace=False)
    visualize_point_clouds_on_sphere([pc_train[i] for i in sample_indices])
    
    # Initialize model
    print("\nInitializing Flow Matching Model...")
    flow_model = RiemannianWassersteinFlowMatching(
        point_clouds=pc_train,
        config=rwfm_config
    )
    
    # Test noise generation
    _, key = test_noise_generation(flow_model, visualize=True)
    
    # Test OT map
    key = test_ot_map(flow_model, key, visualize=True)
    
    # Train model
    train_model(flow_model, batch_size=32, training_steps=200000, decay_steps=2000)
    
    # Generate samples
    _, _, key = generate_and_visualize_samples(flow_model, key, num_samples=16, timesteps=1000)
    
    # Benchmark
    nna, cd_matrix, key = benchmark_model(flow_model, pc_test, key, batch_size=128)
    
    print("\n" + "=" * 80)
    print(f"Final Results - Nearest Neighbor Accuracy: {nna:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
