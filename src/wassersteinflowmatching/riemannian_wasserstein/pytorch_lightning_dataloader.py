"""
PyTorch Lightning DataLoader integration for JAX-based Riemannian Wasserstein Flow Matching.

This module enables using PyTorch DataLoaders with JAX models and supports multi-GPU training.
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from tqdm import trange
from typing import Optional, Callable, Any, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp


class PointCloudDataset(Dataset):
    """PyTorch Dataset for point clouds."""
    
    def __init__(self, point_clouds, weights, conditioning=None):
        """
        Args:
            point_clouds: Array of shape (N, max_points, dim)
            weights: Array of shape (N, max_points)
            conditioning: Optional array of shape (N, cond_dim)
        """
        self.point_clouds = point_clouds
        self.weights = weights
        self.conditioning = conditioning
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        sample = {
            'point_cloud': self.point_clouds[idx],
            'weights': self.weights[idx],
        }
        if self.conditioning is not None:
            sample['conditioning'] = self.conditioning[idx]
        return sample


def numpy_collate(batch):
    """Collate function that returns numpy arrays instead of torch tensors."""
    if isinstance(batch[0], dict):
        return {key: np.stack([item[key] for item in batch]) for key in batch[0]}
    return np.stack(batch)


class MultiGPUTrainer:
    """
    Trainer class that integrates PyTorch Lightning DataLoader with JAX multi-GPU training.
    """
    
    def __init__(
        self,
        flow_model,
        num_devices: Optional[int] = None,
        sharding_strategy: str = 'data_parallel'
    ):
        """
        Args:
            flow_model: RiemannianWassersteinFlowMatching instance
            num_devices: Number of GPUs to use (None = all available)
            sharding_strategy: 'data_parallel' or 'model_parallel'
        """
        self.flow_model = flow_model
        
        # Setup multi-GPU sharding
        self.devices = jax.devices()
        if num_devices is not None:
            self.devices = self.devices[:num_devices]
        
        self.num_devices = len(self.devices)
        print(f"Using {self.num_devices} devices: {self.devices}")
        
        self.sharding_strategy = sharding_strategy
        self._setup_sharding()
        
    def _setup_sharding(self):
        """Setup JAX sharding for multi-GPU training."""
        if self.sharding_strategy == 'data_parallel':
            # Data parallel: replicate model, shard batch across devices
            self.mesh = Mesh(np.array(self.devices), ('data',))
            # Shard first dimension (batch) across devices
            self.data_sharding = NamedSharding(self.mesh, P('data', None, None))
            self.weights_sharding = NamedSharding(self.mesh, P('data', None))
            self.keys_sharding = NamedSharding(self.mesh, P('data', None))  # For random keys (2D)
            # Replicate model parameters across all devices
            self.param_sharding = NamedSharding(self.mesh, P())
        else:
            raise NotImplementedError(f"Sharding strategy {self.sharding_strategy} not implemented")
    
    def shard_batch(self, batch_dict):
        """Shard batch data across devices."""
        sharded_batch = {}
        for key, value in batch_dict.items():
            if key in ['point_cloud']:
                # Shard point clouds
                sharded_batch[key] = jax.device_put(value, self.data_sharding)
            elif key in ['weights']:
                # Shard weights
                sharded_batch[key] = jax.device_put(value, self.weights_sharding)
            elif key in ['conditioning']:
                # Shard conditioning (if present)
                sharded_batch[key] = jax.device_put(value, self.weights_sharding)
            else:
                # Replicate other data
                sharded_batch[key] = jax.device_put(value, self.param_sharding)
        return sharded_batch
    
    def create_dataloader(
        self,
        point_clouds,
        weights,
        conditioning=None,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        multiprocessing_context='spawn'
    ):
        """
        Create a PyTorch DataLoader for the point clouds.
        
        Args:
            point_clouds: Array of point clouds
            weights: Array of weights
            conditioning: Optional conditioning data
            batch_size: Batch size per device (total batch = batch_size * num_devices)
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading
            persistent_workers: Keep workers alive between epochs
            pin_memory: Pin memory for faster GPU transfer
            multiprocessing_context: Multiprocessing context ('spawn', 'fork', 'forkserver', or None).
                                    - 'spawn': JAX-compatible, slower startup (default, safest)
                                    - 'fork': Faster, but may cause issues with JAX (use with caution)
                                    - 'forkserver': Compromise between spawn and fork
                                    - None: Use system default (usually 'fork' on Linux)
            
        Returns:
            PyTorch DataLoader
            
        Note:
            Using 'fork' with JAX may cause warnings or deadlocks. Only use if:
            1. You call multiprocessing.set_start_method('forkserver') BEFORE importing JAX, OR
            2. You're okay with the warning and it doesn't affect your training
        """
        # Adjust batch size for multi-GPU
        # Each device gets batch_size // num_devices samples
        # So we need to ensure batch_size is divisible by num_devices
        if batch_size % self.num_devices != 0:
            adjusted_batch_size = (batch_size // self.num_devices + 1) * self.num_devices
            print(f"Adjusting batch size from {batch_size} to {adjusted_batch_size} "
                  f"to be divisible by {self.num_devices} devices")
            batch_size = adjusted_batch_size
        
        dataset = PointCloudDataset(point_clouds, weights, conditioning)
        
        # Configure multiprocessing context
        mp_context = None
        if num_workers > 0 and multiprocessing_context is not None:
            if multiprocessing_context == 'spawn':
                mp_context = mp.get_context('spawn')
                print(f"Using 'spawn' multiprocessing context (safe with JAX, slower startup)")
            elif multiprocessing_context == 'fork':
                mp_context = mp.get_context('fork')
                print(f"Using 'fork' multiprocessing context (faster, may show JAX warnings)")
            elif multiprocessing_context == 'forkserver':
                mp_context = mp.get_context('forkserver')
                print(f"Using 'forkserver' multiprocessing context (compromise)")
            # None uses system default
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            pin_memory=pin_memory,
            drop_last=True,  # Drop last incomplete batch
            multiprocessing_context=mp_context
        )
        
        return dataloader
    
    def train_step_multi_gpu(self, state, batch_dict, key):
        """
        Single training step on a single GPU (will be pmapped across GPUs).
        
        Args:
            state: Training state for this device
            batch_dict: Batch dictionary with point_clouds, weights, conditioning (already per-device)
            key: Random key for this device
            
        Returns:
            Updated state and loss for this device
        """
        point_clouds_batch = batch_dict['point_cloud']
        weights_batch = batch_dict['weights']
        conditioning_batch = batch_dict.get('conditioning', None)
        
        # Handle CFG null conditioning if enabled
        if hasattr(self.flow_model, 'cfg') and self.flow_model.cfg and conditioning_batch is not None:
            subkey, key = random.split(key)
            batch_size = point_clouds_batch.shape[0]
            is_null_conditioning = random.bernoulli(
                subkey, p=self.flow_model.p_cfg_null, shape=(batch_size,)
            )
        else:
            is_null_conditioning = None
        
        # Generate noise if not using matched noise
        if hasattr(self.flow_model, 'matched_noise') and self.flow_model.matched_noise:
            # TODO: Handle matched noise case
            noise_samples = None
            noise_weights = None
        else:
            noise_samples = None
            noise_weights = None
        
        # Call the original train_step
        state, loss = self.flow_model.train_step(
            state,
            point_clouds_batch,
            weights_batch,
            conditioning_batch,
            noise_samples,
            noise_weights,
            is_null_conditioning,
            key=key
        )
        
        return state, loss
    
    def train_with_dataloader(
        self,
        dataloader: DataLoader,
        training_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        verbose: int = 10,
        learning_rate: float = 2e-4,
        decay_steps: int = 1000,
        saved_state=None,
        key=random.PRNGKey(0),
        checkpoint_callback: Optional[Callable] = None,
        checkpoint_every: int = 1000
    ):
        """
        Train the model using a PyTorch DataLoader on multiple GPUs.
        
        Args:
            dataloader: PyTorch DataLoader
            training_steps: Total number of training steps (overrides epochs if set)
            epochs: Number of epochs to train
            verbose: Print loss every N steps
            learning_rate: Learning rate
            decay_steps: Learning rate decay steps
            saved_state: Optional saved state to resume from
            key: Random key
            checkpoint_callback: Function to call for checkpointing
            checkpoint_every: Save checkpoint every N steps
        """
        if training_steps is None and epochs is None:
            raise ValueError("Must specify either training_steps or epochs")
        
        # Initialize training state
        subkey, key = random.split(key)
        
        if saved_state is None:
            self.flow_model.state = self.flow_model.create_train_state(
                model=self.flow_model.FlowMatchingModel,
                learning_rate=learning_rate,
                decay_steps=decay_steps,
                key=subkey
            )
        else:
            self.flow_model.state = saved_state
            print(f"Resuming training from step {int(self.flow_model.state.step)}")
        
        # Get initial step before replicating
        initial_step = int(self.flow_model.state.step)
        
        # Replicate state across devices by stacking
        state = jax.tree.map(
            lambda x: jnp.stack([x] * self.num_devices),
            self.flow_model.state
        )
        
        # Create pmapped training step
        train_step_pmapped = jax.pmap(
            self.train_step_multi_gpu,
            axis_name='data',
            devices=self.devices
        )
        
        # Training loop
        step = initial_step
        losses = []
        
        if training_steps is not None:
            total_steps = training_steps
            num_epochs = (training_steps // len(dataloader)) + 1
        else:
            num_epochs = epochs
            total_steps = num_epochs * len(dataloader)
        
        print(f"Training for {total_steps} steps across {num_epochs} epochs")
        print(f"Batch size per device: {dataloader.batch_size // self.num_devices}")
        print(f"Total batch size: {dataloader.batch_size}")
        
        try:
            pbar = trange(total_steps, desc="Training", initial=step)
            
            for epoch in range(num_epochs):
                for batch_dict in dataloader:
                    if step >= total_steps:
                        break
                    
                    # Convert numpy arrays to JAX arrays
                    batch_dict = {k: jnp.array(v) for k, v in batch_dict.items()}
                    
                    # Reshape batch to (num_devices, batch_per_device, ...)
                    per_device_batch_size = dataloader.batch_size // self.num_devices
                    reshaped_batch = {}
                    for batch_key, value in batch_dict.items():
                        # Reshape from (total_batch, ...) to (num_devices, batch_per_device, ...)
                        if batch_key == 'point_cloud':
                            reshaped_batch[batch_key] = value.reshape(
                                self.num_devices, per_device_batch_size, *value.shape[1:]
                            )
                        elif batch_key in ['weights', 'conditioning']:
                            reshaped_batch[batch_key] = value.reshape(
                                self.num_devices, per_device_batch_size, *value.shape[1:]
                            )
                        else:
                            reshaped_batch[batch_key] = value
                    
                    # Split key for this step
                    subkey, key = random.split(key)
                    # Split keys across devices
                    keys = random.split(subkey, self.num_devices)
                    
                    # Training step
                    state, loss = train_step_pmapped(state, reshaped_batch, keys)
                    
                    # Aggregate loss from all devices (take mean)
                    loss = jnp.mean(loss)
                    losses.append(float(loss))
                    
                    # Update progress bar
                    if step % verbose == 0:
                        pbar.set_postfix({'loss': f'{loss:.6f}', 'epoch': epoch + 1})
                    
                    pbar.update(1)
                    step += 1
                    
                    # Checkpoint
                    if checkpoint_callback is not None and step % checkpoint_every == 0:
                        # Get state from first device for checkpointing
                        checkpoint_state = jax.tree.map(lambda x: x[0], state)
                        checkpoint_callback(checkpoint_state, step, losses)
                    
                    if step >= total_steps:
                        break
            
            pbar.close()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        # Save final state (get from first device)
        self.flow_model.state = jax.tree.map(lambda x: x[0], state)
        self.flow_model.params = self.flow_model.state.params
        self.flow_model.losses = losses
        
        print(f"\nTraining completed. Final loss: {losses[-1]:.6f}")
        
        return losses


def create_trainer_from_flow_model(
    flow_model,
    batch_size=32,
    num_workers=4,
    num_devices=None,
    shuffle=True
):
    """
    Convenience function to create a multi-GPU trainer with DataLoader.
    
    Args:
        flow_model: RiemannianWassersteinFlowMatching instance
        batch_size: Batch size (will be distributed across GPUs)
        num_workers: Number of DataLoader workers
        num_devices: Number of GPUs to use (None = all)
        shuffle: Whether to shuffle training data
        
    Returns:
        MultiGPUTrainer instance and DataLoader
    """
    trainer = MultiGPUTrainer(flow_model, num_devices=num_devices)
    
    dataloader = trainer.create_dataloader(
        point_clouds=flow_model.point_clouds,
        weights=flow_model.weights,
        conditioning=flow_model.conditioning if hasattr(flow_model, 'conditioning') else None,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return trainer, dataloader
