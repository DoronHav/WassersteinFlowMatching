import pytest
import numpy as np
import jax.numpy as jnp
from jax import random
import anndata
import pandas as pd

# Import the class to be tested
from wassersteinflowmatching.wasserstein import SpatialWassersteinFlowMatching
from wassersteinflowmatching.wasserstein.DefaultConfig import SpatialDefaultConfig

# --- Test Data Fixtures for Spatial Data ---

@pytest.fixture
def unconditional_adata():
    """Provides a simple AnnData object for unconditional spatial model testing."""
    num_cells = 100
    num_genes = 50
    
    # Create expression data
    X = np.random.randn(num_cells, num_genes).astype(np.float32)
    
    # Create spatial coordinates
    spatial_coords = np.random.rand(num_cells, 2) * 100
    
    # Create the AnnData object
    adata = anndata.AnnData(X=X)
    adata.obsm['spatial'] = spatial_coords
    
    return adata

@pytest.fixture
def conditional_adata():
    """
    Provides an AnnData object with categorical and continuous data
    for testing conditional model setup.
    """
    num_cells = 100
    num_genes = 50
    
    # Create expression data
    X = np.random.randn(num_cells, num_genes).astype(np.float32)
    
    # Create spatial coordinates
    spatial_coords = np.random.rand(num_cells, 2) * 100
    
    # Create categorical observation data (e.g., cell types)
    cell_types = np.random.choice(['Tumor', 'Stroma', 'Immune'], size=num_cells)
    obs_df = pd.DataFrame({'cell_type': pd.Series(cell_types, dtype='category')})
    
    # Create continuous embedding data (e.g., PCA)
    pca_embedding = np.random.randn(num_cells, 10).astype(np.float32)
    
    # Create the AnnData object
    adata = anndata.AnnData(X=X, obs=obs_df)
    adata.obsm['spatial'] = spatial_coords
    adata.obsm['X_pca'] = pca_embedding
    
    return adata

# --- Core Functionality Tests ---

def test_model_initialization(unconditional_adata):
    """Tests if the spatial model initializes without errors."""
    try:
        model = SpatialWassersteinFlowMatching(adata=unconditional_adata, k_neighbours=5)
        assert model is not None
        assert model.space_dim == unconditional_adata.n_vars
        assert model.max_niche_size == 5
        assert model.conditioning_vectors is None
    except Exception as e:
        pytest.fail(f"Default spatial model initialization failed: {e}")

def test_conditional_model_initialization(conditional_adata):
    """Tests if the model correctly processes conditioning information."""
    conditioning_obs = ['cell_type']
    conditioning_obsm = ['X_pca']
    
    model = SpatialWassersteinFlowMatching(
        adata=conditional_adata, 
        k_neighbours=5,
        conditioning_obs=conditioning_obs,
        conditioning_obsm=conditioning_obsm
    )
    
    assert model.conditioning_vectors is not None
    # Expected shape: 3 (one-hot from 'cell_type') + 10 (from 'X_pca') = 13 features
    expected_features = 3 + 10
    assert model.conditioning_vectors.shape == (conditional_adata.n_obs, expected_features)
    assert model.guidance_gamma > 0 # Should be set from config

# --- Training and Generation Tests ---

def test_unconditional_train_and_generate(unconditional_adata):
    """
    Tests a full train and generation cycle for an unconditional spatial model.
    """
    model = SpatialWassersteinFlowMatching(adata=unconditional_adata, k_neighbours=8)

    # Test training
    model.train(training_steps=10, batch_size=4, decay_steps=5, verbose=2)
    assert model.state.step == 10
    final_loss = model.losses[-1]
    assert np.isfinite(final_loss), "Loss should be a finite number."

    # Test generation
    num_samples = 4
    timesteps = 8
    generated_trajectory, gen_labels = model.generate_niches(
        num_samples=num_samples,
        timesteps=timesteps
    )

    assert isinstance(generated_trajectory, list)
    assert len(generated_trajectory) == timesteps + 1
    final_niches = generated_trajectory[-1]
    
    assert isinstance(final_niches, jnp.ndarray)
    assert final_niches.shape == (num_samples, model.max_niche_size, model.space_dim)
    assert not np.any(np.isnan(final_niches)), "Generated niches should not contain NaNs."
    assert gen_labels is None, "Labels should be None for an unconditional model."

def test_conditional_train_and_generate(conditional_adata):
    """
    Tests the full cycle for a spatial model with conditioning and guidance.
    """
    model = SpatialWassersteinFlowMatching(
        adata=conditional_adata, 
        k_neighbours=8,
        conditioning_obs=['cell_type'],
        conditioning_obsm=['X_pca'],
        config=SpatialDefaultConfig(guidance_gamma=1.5, p_uncond=0.1)
    )
    
    # Test training
    model.train(training_steps=10, batch_size=4, decay_steps=5)
    assert model.state.step == 10
    assert np.isfinite(model.losses[-1])

    # Test generation with a specific condition
    num_samples = 4
    timesteps = 8
    
    # Create a sample condition vector to generate for
    # (e.g., 'Tumor' type and a random PCA embedding)
    sample_condition = np.concatenate([
        np.array([0., 0., 1.]), # one-hot for 'Tumor' (assuming this order)
        np.random.randn(10)
    ])
    
    generated_trajectory, gen_labels = model.generate_niches(
        num_samples=num_samples,
        timesteps=timesteps,
        conditioning_info=sample_condition
    )

    assert len(generated_trajectory) == timesteps + 1
    final_niches = generated_trajectory[-1]
    assert final_niches.shape == (num_samples, model.max_niche_size, model.space_dim)
    assert not np.any(np.isnan(final_niches))
    
    # Check that the generated labels match the input condition
    assert gen_labels is not None
    assert gen_labels.shape == (num_samples, sample_condition.shape[0])
    assert np.allclose(gen_labels, np.tile(sample_condition, (num_samples, 1)))
