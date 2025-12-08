#!/usr/bin/env python3
"""
Example: Training Riemannian Wasserstein Flow Matching with PyTorch DataLoader and Multi-GPU support.

This script demonstrates how to use PyTorch Lightning DataLoaders with JAX-based RWFM
for efficient multi-GPU training.

Note: The DataLoader automatically uses 'spawn' multiprocessing context which is JAX-compatible.
No need to set global multiprocessing methods.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Set which GPUs to use

import jax
import numpy as np
import matplotlib.pyplot as plt
from wassersteinflowmatching.riemannian_wasserstein import RiemannianWassersteinFlowMatching
from wassersteinflowmatching.riemannian_wasserstein.pytorch_lightning_dataloader import (
    create_trainer_from_flow_model,
    MultiGPUTrainer
)


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


def checkpoint_callback(state, step, losses):
    """Example checkpoint callback function."""
    print(f"\n[Checkpoint] Step {step}, Recent loss: {losses[-1]:.6f}")
    # You can add actual checkpoint saving logic here
    # e.g., with pickle or flax.serialization
    # with open(f'checkpoint_step_{step}.pkl', 'wb') as f:
    #     pickle.dump({'state': state, 'step': step, 'losses': losses}, f)


def example_basic_multi_gpu_training(pc_train):
    """
    Example 1: Basic multi-GPU training with automatic setup.
    """
    print("\n" + "="*80)
    print("Example 1: Basic Multi-GPU Training")
    print("="*80)
    
    # Initialize the flow model
    print("\nInitializing Flow Matching Model...")
    flow_model = RiemannianWassersteinFlowMatching(
        point_clouds=pc_train,
        config=rwfm_config
    )
    
    # Create trainer and dataloader (automatically uses all GPUs)
    print("\nCreating Multi-GPU Trainer and DataLoader...")
    trainer = MultiGPUTrainer(flow_model=flow_model, num_devices=None)
    
    dataloader = trainer.create_dataloader(
        point_clouds=flow_model.point_clouds,
        weights=flow_model.weights,
        batch_size=64,  # Total batch size (will be split across GPUs)
        num_workers=8,   # PyTorch DataLoader workers
        shuffle=True,
        multiprocessing_context='fork'  # Fastest data loading
    )
    
    print(f"DataLoader created with {len(dataloader)} batches per epoch")
    print(f"Using {trainer.num_devices} GPUs")
    
    # Train the model
    print("\nStarting training...")
    losses = trainer.train_with_dataloader(
        dataloader=dataloader,
        epochs=10,  # Train for 10 epochs
        verbose=10,  # Print loss every 10 steps
        learning_rate=2e-4,
        decay_steps=2000,
        checkpoint_callback=checkpoint_callback,
        checkpoint_every=500
    )
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    loss_smooth = np.convolve(np.log(losses), np.ones(100) / 100, mode='valid')
    plt.plot(loss_smooth)
    plt.xlabel('Training Step')
    plt.ylabel('Log Loss (smoothed)')
    plt.title('Multi-GPU Training Loss Curve')
    plt.savefig('multi_gpu_training_loss.png')
    plt.show()
    
    return flow_model, losses


def example_custom_multi_gpu_setup(pc_train):
    """
    Example 2: Custom multi-GPU setup with more control.
    """
    print("\n" + "="*80)
    print("Example 2: Custom Multi-GPU Setup")
    print("="*80)
    
    # Initialize the flow model
    flow_model = RiemannianWassersteinFlowMatching(
        point_clouds=pc_train,
        config=rwfm_config
    )
    
    # Create trainer and dataloader with 'fork' context
    trainer = MultiGPUTrainer(flow_model=flow_model, num_devices=2)
    
    dataloader = trainer.create_dataloader(
        point_clouds=flow_model.point_clouds,
        weights=flow_model.weights,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        multiprocessing_context='fork'  # Fastest data loading
    )
    
    print(f"Using {trainer.num_devices} GPUs: {trainer.devices}")
    
    # Train with specific number of steps
    losses = trainer.train_with_dataloader(
        dataloader=dataloader,
        training_steps=5000,  # Train for specific number of steps
        verbose=50,
        learning_rate=1e-4,
        decay_steps=1000
    )
    
    return flow_model, losses


def example_with_conditioning(pc_train, conditioning_data):
    """
    Example 3: Multi-GPU training with conditioning.
    """
    print("\n" + "="*80)
    print("Example 3: Multi-GPU Training with Conditioning")
    print("="*80)
    
    # Initialize the flow model with conditioning
    flow_model = RiemannianWassersteinFlowMatching(
        point_clouds=pc_train,
        conditioning=conditioning_data,
        config=rwfm_config
    )
    
    # Create trainer and dataloader
    trainer = MultiGPUTrainer(flow_model=flow_model, num_devices=None)
    
    dataloader = trainer.create_dataloader(
        point_clouds=flow_model.point_clouds,
        weights=flow_model.weights,
        conditioning=flow_model.conditioning,
        batch_size=64,
        shuffle=True,
        num_workers=8
    )
    
    # Train
    losses = trainer.train_with_dataloader(
        dataloader=dataloader,
        epochs=5,
        verbose=20,
        learning_rate=2e-4,
        decay_steps=2000
    )
    
    return flow_model, losses


def example_resume_training(pc_train, saved_state):
    """
    Example 4: Resume training from checkpoint.
    """
    print("\n" + "="*80)
    print("Example 4: Resume Training from Checkpoint")
    print("="*80)
    
    # Initialize the flow model
    flow_model = RiemannianWassersteinFlowMatching(
        point_clouds=pc_train,
        config=rwfm_config
    )
    
    # Create trainer and dataloader
    trainer = MultiGPUTrainer(flow_model=flow_model, num_devices=None)
    
    dataloader = trainer.create_dataloader(
        point_clouds=flow_model.point_clouds,
        weights=flow_model.weights,
        conditioning=flow_model.conditioning,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        multiprocessing_context='fork'
    )
    
    # Resume training from saved state
    losses = trainer.train_with_dataloader(
        dataloader=dataloader,
        epochs=10,
        saved_state=saved_state,  # Pass the saved state
        verbose=10,
        learning_rate=1e-4,  # Can use different learning rate
        decay_steps=2000
    )
    
    return flow_model, losses


def example_with_fork_dataloader(pc_train):
    """
    Example: Using 'fork' for faster data loading (with JAX warning suppression).
    
    Note: 'fork' is faster than 'spawn' but may cause warnings with JAX.
    The warning is usually harmless if you set forkserver before importing JAX.
    """
    print("\n" + "="*80)
    print("Example: Using 'fork' for Faster Data Loading")
    print("="*80)
    
    # Initialize the flow model
    flow_model = RiemannianWassersteinFlowMatching(
        point_clouds=pc_train,
        config=rwfm_config
    )
    
    # Create trainer and dataloader with 'fork' context
    trainer = MultiGPUTrainer(flow_model=flow_model, num_devices=None)
    
    dataloader = trainer.create_dataloader(
        point_clouds=flow_model.point_clouds,
        weights=flow_model.weights,
        conditioning=None,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        multiprocessing_context='fork'  # Faster, but may show warning
    )
    
    print("Note: You may see a fork() warning. This is usually harmless.")
    print("To avoid the warning, add this at the TOP of your script (before importing jax):")
    print("  import multiprocessing")
    print("  multiprocessing.set_start_method('forkserver')")
    
    # Train
    losses = trainer.train_with_dataloader(
        dataloader=dataloader,
        epochs=5,
        verbose=20,
        learning_rate=2e-4,
        decay_steps=2000
    )
    
    return flow_model, losses


def generate_dummy_spherical_data(num_samples=1000, num_points=150):
    """Generate dummy spherical data for testing."""
    print("Generating dummy spherical point cloud data...")
    
    # Generate random points on sphere
    point_clouds = []
    for _ in range(num_samples):
        # Random points on unit sphere
        points = np.random.randn(num_points, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        # Add some structure (cluster around a direction)
        center = np.random.randn(3)
        center = center / np.linalg.norm(center)
        noise = np.random.randn(num_points, 3) * 0.3
        points = points + noise + center
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        point_clouds.append(points)
    
    return point_clouds


def main():
    """Main execution function."""
    print("="*80)
    print("PyTorch DataLoader + JAX Multi-GPU Training Examples")
    print("="*80)
    
    # Check available devices
    print(f"\nAvailable JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Generate or load data
    # For this example, we'll generate dummy data
    # Replace this with your actual data loading
    pc_train = generate_dummy_spherical_data(num_samples=1000, num_points=150)
    print(f"Training samples: {len(pc_train)}")
    
    # Run Example 1: Basic multi-GPU training
    flow_model, losses = example_basic_multi_gpu_training(pc_train)
    
    # Optionally run other examples:
    
    # Example 2: Custom setup
    # flow_model, losses = example_custom_multi_gpu_setup(pc_train)
    
    # Example 3: With conditioning
    # conditioning_data = np.random.randn(len(pc_train), 10)  # Dummy conditioning
    # flow_model, losses = example_with_conditioning(pc_train, conditioning_data)
    
    # Example 4: Resume training
    # saved_state = flow_model.state  # From previous training
    # flow_model, losses = example_resume_training(pc_train, saved_state)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    # Generate samples to verify the model works
    print("\nGenerating samples...")
    key = jax.random.PRNGKey(42)
    generated_samples, sample_weights = flow_model.generate_samples(
        num_samples=16,
        timesteps=100,
        key=key
    )
    print(f"Generated {len(generated_samples[-1])} samples")


if __name__ == "__main__":
    main()
