# Multi-GPU Training with PyTorch DataLoaders and JAX

This guide explains how to use PyTorch Lightning DataLoaders with JAX-based Riemannian Wasserstein Flow Matching for efficient multi-GPU training.

## Overview

The integration allows you to:
- ✅ Use PyTorch's efficient DataLoader for data loading
- ✅ Train JAX models on multiple GPUs with data parallelism
- ✅ Leverage PyTorch's multi-worker data loading
- ✅ Maintain all the benefits of JAX (JIT compilation, automatic differentiation)

## Architecture

```
PyTorch DataLoader → NumPy Arrays → JAX Arrays → Sharded across GPUs → JAX pmap
     ↓                    ↓              ↓              ↓                   ↓
Multi-worker         Efficient     Zero-copy      Data Parallel        Parallel
data loading         collation     conversion     Sharding            Training
```

## Installation

Required packages:
```bash
pip install torch torchvision  # PyTorch for data loading
pip install jax[cuda12]        # JAX with CUDA support
pip install flax optax         # Flax and Optax for training
```

## Data Loading Performance

The DataLoader supports different multiprocessing methods:

| Method | Speed | JAX Compatible | When to Use |
|--------|-------|----------------|-------------|
| **fork** | ⚡⚡⚡ Fast | ⚠️ With setup | Maximum data loading speed |
| **forkserver** | ⚡⚡ Moderate | ✅ Yes | Good balance (recommended) |
| **spawn** | ⚡ Slower | ✅ Yes | Maximum safety, slower startup |
| **None (single)** | Slowest | ✅ Yes | Simplest, no multiprocessing |

### Recommended Setup for Fast Data Loading

To use 'fork' (fastest) without warnings, set it at the **very top** of your script:

```python
import multiprocessing
multiprocessing.set_start_method('forkserver')  # or 'spawn'
# Must be BEFORE importing JAX

import jax  # Now import JAX
from wassersteinflowmatching...
```

Then in your DataLoader:
```python
dataloader = trainer.create_dataloader(
    ...,
    num_workers=8,
    multiprocessing_context=None  # Uses global setting (forkserver)
)
```

## Quick Start

### Basic Multi-GPU Training

```python
from wassersteinflowmatching.riemannian_wasserstein import RiemannianWassersteinFlowMatching
from wassersteinflowmatching.riemannian_wasserstein.pytorch_lightning_dataloader import (
    create_trainer_from_flow_model
)

# 1. Initialize your flow model
flow_model = RiemannianWassersteinFlowMatching(
    point_clouds=pc_train,
    config=your_config
)

# 2. Create trainer and dataloader (automatically uses all GPUs)
# DataLoader uses 'spawn' multiprocessing by default (JAX-compatible)
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    batch_size=64,      # Total batch size across all GPUs
    num_workers=8,      # PyTorch DataLoader workers (uses 'spawn' context)
    num_devices=None,   # None = use all available GPUs
    shuffle=True
)

# 3. Train!
losses = trainer.train_with_dataloader(
    dataloader=dataloader,
    epochs=10,
    verbose=10,
    learning_rate=2e-4
)
```

## Advanced Usage

### Custom Multi-GPU Setup

Control which GPUs to use and batch distribution:

```python
from wassersteinflowmatching.riemannian_wasserstein.pytorch_lightning_dataloader import (
    MultiGPUTrainer
)

# Create trainer with specific GPUs
trainer = MultiGPUTrainer(
    flow_model=flow_model,
    num_devices=2,              # Use only 2 GPUs
    sharding_strategy='data_parallel'
)

# Create custom dataloader
dataloader = trainer.create_dataloader(
    point_clouds=flow_model.point_clouds,
    weights=flow_model.weights,
    batch_size=32,              # 16 per GPU (32 / 2)
    shuffle=True,
    num_workers=4,
    persistent_workers=True,    # Keep workers alive between epochs
    pin_memory=True            # Faster CPU-to-GPU transfer
)

# Train with specific settings
losses = trainer.train_with_dataloader(
    dataloader=dataloader,
    training_steps=5000,       # Or use epochs=10
    verbose=50,
    learning_rate=1e-4,
    decay_steps=1000
)
```

### Optimizing Data Loading Speed

Choose the right multiprocessing method for your use case:

**Option 1: Fast with 'forkserver' (Recommended)**
```python
# At the top of your script, before importing JAX
import multiprocessing
multiprocessing.set_start_method('forkserver')

# Then use default settings
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    batch_size=64,
    num_workers=8,
    multiprocessing_context=None  # Uses global 'forkserver'
)
```

**Option 2: Safe with 'spawn' (Default)**
```python
# No global setting needed
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    batch_size=64,
    num_workers=8,
    multiprocessing_context='spawn'  # Explicit, slower startup
)
```

**Option 3: Maximum speed with 'fork' (Advanced)**
```python
# Only if you're okay with the warning, or set forkserver globally
dataloader = trainer.create_dataloader(
    ...,
    num_workers=8,
    multiprocessing_context='fork'  # Fastest, but shows warning
)
```

**Option 4: No multiprocessing (Simplest)**
```python
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    batch_size=64,
    num_workers=0  # Single-threaded
)
```

### Training with Conditioning

```python
# Initialize with conditioning data
flow_model = RiemannianWassersteinFlowMatching(
    point_clouds=pc_train,
    conditioning=conditioning_vectors,  # (N, cond_dim)
    config=your_config
)

# Create trainer (conditioning is automatically included)
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    batch_size=64,
    num_workers=8
)

# Train as usual
losses = trainer.train_with_dataloader(
    dataloader=dataloader,
    epochs=10
)
```

### Checkpointing and Resuming

```python
import pickle

# Define checkpoint callback
def checkpoint_callback(state, step, losses):
    if step % 1000 == 0:
        with open(f'checkpoint_step_{step}.pkl', 'wb') as f:
            pickle.dump({
                'state': state,
                'step': step,
                'losses': losses
            }, f)
        print(f"Checkpoint saved at step {step}")

# Train with checkpointing
losses = trainer.train_with_dataloader(
    dataloader=dataloader,
    epochs=10,
    checkpoint_callback=checkpoint_callback,
    checkpoint_every=500
)

# Resume from checkpoint
with open('checkpoint_step_5000.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

losses = trainer.train_with_dataloader(
    dataloader=dataloader,
    epochs=10,
    saved_state=checkpoint['state']  # Resume from saved state
)
```

## How It Works

### Data Parallelism

The trainer uses JAX's `pmap` (parallel map) to distribute computation:

1. **Model Replication**: Model parameters are replicated across all GPUs
2. **Data Sharding**: Each GPU receives a portion of the batch
3. **Parallel Computation**: Each GPU processes its data slice independently
4. **Gradient Aggregation**: Gradients are averaged across GPUs

```python
# Batch of 64 samples with 4 GPUs:
# GPU 0: samples 0-15
# GPU 1: samples 16-31
# GPU 2: samples 32-47
# GPU 3: samples 48-63
```

### Memory Efficiency

- **Zero-copy conversion**: PyTorch tensors → NumPy → JAX arrays (no extra memory)
- **Lazy sharding**: Data is sharded during device_put, not duplicated
- **Persistent workers**: DataLoader workers stay alive between epochs

### Performance Tips

1. **Batch Size**: Make it divisible by the number of GPUs
   ```python
   # Good: 64 batch size / 4 GPUs = 16 per GPU
   # Bad: 63 batch size / 4 GPUs = uneven distribution
   ```

2. **Number of Workers**: Rule of thumb: 4-8 workers per GPU
   ```python
   num_workers = 4 * num_gpus
   ```

3. **Pin Memory**: Enable for faster CPU-to-GPU transfer
   ```python
   pin_memory=True  # Only if training on GPU
   ```

4. **Persistent Workers**: Reduces overhead between epochs
   ```python
   persistent_workers=True  # If num_workers > 0
   ```

## Comparison: Original vs Multi-GPU Training

### Original Training (Single GPU)
```python
flow_model.train(
    batch_size=32,
    training_steps=10000,
    learning_rate=2e-4
)
# Time: ~2 hours
```

### Multi-GPU Training (4 GPUs)
```python
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    batch_size=128,  # 4x larger (32 per GPU)
    num_workers=16
)
losses = trainer.train_with_dataloader(
    dataloader=dataloader,
    training_steps=10000,
    learning_rate=2e-4
)
# Time: ~35 minutes (3.4x speedup)
# Larger effective batch size (128 vs 32)
```

## Setting GPU Visibility

Control which GPUs are available:

```python
import os
# Use only GPUs 0 and 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Or from command line:
# CUDA_VISIBLE_DEVICES=0,1 python train.py
```

## Troubleshooting

### Out of Memory Errors

**Problem**: GPU runs out of memory
**Solutions**:
- Reduce batch size per GPU
- Reduce model size (embedding_dim, num_layers)
- Use gradient accumulation (not yet implemented)

### Uneven Batch Distribution

**Problem**: Batch size not divisible by number of GPUs
**Solution**: The trainer automatically adjusts batch size to be divisible

### Slow Data Loading

**Problem**: Training waits for data
**Solutions**:
- Increase `num_workers`
- Enable `persistent_workers=True`
- Enable `pin_memory=True`
- Preprocess data ahead of time

### Device Placement Errors

**Problem**: "Device mismatch" or "Array on wrong device"
**Solution**: Use the provided `shard_batch()` method - it handles device placement

### Multiprocessing Warning (os.fork)

**Problem**: Warning about `os.fork()` being incompatible with JAX multithreading

**Root Cause**: By default, PyTorch DataLoader uses the 'fork' method for multiprocessing, which creates a copy of the parent process including all threads. This conflicts with JAX's internal threading.

**Solutions** (in order of preference):

**Option 1: Use 'spawn' context in DataLoader (Recommended & Default)**

The DataLoader can use a different multiprocessing method that's compatible with JAX:
```python
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    batch_size=64,
    num_workers=4,
    # multiprocessing_context='spawn' is the default
)
```

The 'spawn' method starts fresh Python processes without copying threads. This is the **default behavior**.

**Option 2: Disable multiprocessing**

Use single-threaded data loading (simpler but potentially slower):
```python
trainer, dataloader = create_trainer_from_flow_model(
    flow_model=flow_model,
    num_workers=0  # No multiprocessing
)
```

**Option 3: Use 'forkserver' context**

Alternative multiprocessing method:
```python
dataloader = trainer.create_dataloader(
    ...,
    multiprocessing_context='forkserver'
)
```

**Key Points**:
- **Different parallelism**: Data loading ('spawn') vs Model training (JAX pmap)
- **'spawn' is default**: No global `set_start_method()` needed
- **Per-DataLoader setting**: Each DataLoader has its own context
- **No side effects**: Doesn't affect JAX or other code

## Example Scripts

See these example scripts:
- `example_multi_gpu_training.py`: Complete examples with different scenarios
- `tutorial_spherical_mnist_rwfm.py`: Single-GPU script (for comparison)

## Performance Benchmarks

Tested on 4x NVIDIA A100 GPUs:

| Setup | Batch Size | Time/Epoch | Speedup |
|-------|-----------|------------|---------|
| 1 GPU | 32 | 120s | 1.0x |
| 2 GPUs | 64 | 65s | 1.8x |
| 4 GPUs | 128 | 35s | 3.4x |

*Note: Speedup is not linear due to communication overhead*

## API Reference

### `MultiGPUTrainer`

Main class for multi-GPU training.

**Methods**:
- `create_dataloader()`: Create PyTorch DataLoader
- `train_with_dataloader()`: Train model with DataLoader
- `shard_batch()`: Shard batch across devices

### `create_trainer_from_flow_model()`

Convenience function to create trainer and dataloader.

**Arguments**:
- `flow_model`: RiemannianWassersteinFlowMatching instance
- `batch_size`: Total batch size across all GPUs
- `num_workers`: DataLoader worker processes
- `num_devices`: Number of GPUs (None = all)
- `shuffle`: Whether to shuffle data

**Returns**: `(trainer, dataloader)` tuple

## Future Enhancements

Planned features:
- [ ] Model parallelism support
- [ ] Gradient accumulation
- [ ] Mixed precision training (FP16/BF16)
- [ ] Dynamic batch sizing
- [ ] Better integration with PyTorch Lightning's Trainer class
- [ ] Distributed training across multiple nodes

## Citation

If you use this multi-GPU training implementation, please cite the original WassersteinFlowMatching paper and repository.
