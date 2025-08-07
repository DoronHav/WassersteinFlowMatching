import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from wassersteinflowmatching.wasserstein import WassersteinFlowMatching
from wassersteinflowmatching.wasserstein.DefaultConfig import DefaultConfig

# --- Test Data Fixtures ---

@pytest.fixture
def unconditional_data():
    """Provides simple, unconditional point cloud data for testing."""
    key = random.key(0)
    point_cloud_sizes = random.randint(key, shape=(32,), minval=10, maxval=20)
    point_clouds = [np.random.randn(n, 3) for n in point_cloud_sizes]
    return {"point_clouds": point_clouds}

@pytest.fixture
def discrete_conditional_data():
    """Provides point clouds with discrete (string) labels."""
    key = random.key(1)
    num_samples = 32
    point_cloud_sizes = random.randint(key, shape=(num_samples,), minval=10, maxval=20)
    point_clouds = [np.random.randn(n, 3) for n in point_cloud_sizes]
    labels = np.random.choice(['cat', 'dog', 'tree'], size=num_samples)
    return {"point_clouds": point_clouds, "labels": labels}

@pytest.fixture
def continuous_conditional_data():
    """Provides point clouds with continuous (vector) labels."""
    key = random.key(2)
    num_samples = 32
    label_dim = 4
    point_cloud_sizes = random.randint(key, shape=(num_samples,), minval=10, maxval=20)
    point_clouds = [np.random.randn(n, 3) for n in point_cloud_sizes]
    labels = np.random.randn(num_samples, label_dim)
    return {"point_clouds": point_clouds, "labels": labels}

@pytest.fixture
def matched_noise_data():
    """Provides a source and a target dataset for matched-pair training."""
    key = random.key(3)
    num_samples = 32
    # Source (noise) point clouds
    noise_pc_sizes = random.randint(key, shape=(num_samples,), minval=10, maxval=20)
    noise_point_clouds = [np.random.uniform(size=(n, 3)) for n in noise_pc_sizes]
    # Target (data) point clouds
    pc_sizes = random.randint(key, shape=(num_samples,), minval=10, maxval=20)
    point_clouds = [np.random.randn(n, 3) * 0.5 + 2 for n in pc_sizes]
    return {"point_clouds": point_clouds, "noise_point_clouds": noise_point_clouds}


# --- Core Functionality Tests ---

def test_model_initialization(unconditional_data):
    """Tests if the model initializes without errors with default config."""
    try:
        model = WassersteinFlowMatching(point_clouds=unconditional_data["point_clouds"])
        assert model is not None
        assert model.point_clouds.shape[0] == 32
        assert model.space_dim == 3
    except Exception as e:
        pytest.fail(f"Default model initialization failed: {e}")

@pytest.mark.parametrize("monge_map_type", ['euclidean', 'entropic', 'sample', 'argmax', 'rounded_matching'])
def test_initialization_with_monge_maps(unconditional_data, monge_map_type):
    """Tests model initialization with different Monge map strategies."""
    try:
        model = WassersteinFlowMatching(
            point_clouds=unconditional_data["point_clouds"],
            config=DefaultConfig(monge_map=monge_map_type)
        )
        assert model.monge_map == monge_map_type
    except Exception as e:
        pytest.fail(f"Initialization with monge_map='{monge_map_type}' failed: {e}")

@pytest.mark.parametrize("noise_type", ['uniform', 'normal', 'chol_normal', 'meta_normal', 'student_t'])
def test_initialization_with_noise_types(unconditional_data, noise_type):
    """Tests model initialization with different noise types."""
    try:
        model = WassersteinFlowMatching(
            point_clouds=unconditional_data["point_clouds"],
            config=DefaultConfig(noise_type=noise_type)
        )
        assert model.noise_type == noise_type
    except Exception as e:
        pytest.fail(f"Initialization with noise_type='{noise_type}' failed: {e}")

# --- Training and Generation Tests ---

def test_unconditional_train_and_generate(unconditional_data):
    """
    Tests a full train and generation cycle for an unconditional model.
    Asserts output shapes and validity.
    """
    model = WassersteinFlowMatching(point_clouds=unconditional_data["point_clouds"])

    # Test training
    model.train(training_steps=10, batch_size=4, decay_steps=5, verbose=2)
    assert model.state.step == 10
    final_loss = model.losses[-1]
    assert np.isfinite(final_loss), "Loss should be a finite number."

    # Test generation
    num_samples = 4
    timesteps = 8
    generated_trajectory, weights = model.generate_samples(
        num_samples=num_samples,
        timesteps=timesteps
    )

    assert isinstance(generated_trajectory, list)
    assert len(generated_trajectory) == timesteps + 1
    final_samples = generated_trajectory[-1]
    assert isinstance(final_samples, jnp.ndarray)
    assert final_samples.shape[0] == num_samples
    assert final_samples.shape[2] == model.space_dim
    assert not np.any(np.isnan(final_samples)), "Generated samples should not contain NaNs."
    assert weights.shape[0] == num_samples

def test_discrete_conditional_train_and_generate(discrete_conditional_data):
    """Tests the full cycle for a model with discrete labels and guidance."""
    model = WassersteinFlowMatching(
        point_clouds=discrete_conditional_data["point_clouds"],
        labels=discrete_conditional_data["labels"],
        config=DefaultConfig(guidance_gamma=1.5, p_uncond=0.1)
    )
    assert model.discrete_labels
    assert model.guidance_gamma > 1.0

    # Test training
    model.train(training_steps=10, batch_size=4, decay_steps=5)
    assert model.state.step == 10
    assert np.isfinite(model.losses[-1])

    # Test generation
    num_samples = 4
    timesteps = 8
    target_label = 'cat'
    generated_trajectory, weights, final_labels = model.generate_samples(
        num_samples=num_samples,
        timesteps=timesteps,
        generate_labels=target_label
    )

    assert len(generated_trajectory) == timesteps + 1
    final_samples = generated_trajectory[-1]
    assert final_samples.shape[0] == num_samples
    assert not np.any(np.isnan(final_samples))
    assert all(label == target_label for label in final_labels)

def test_continuous_conditional_train_and_generate(continuous_conditional_data):
    """Tests the full cycle for a model with continuous labels and guidance."""
    model = WassersteinFlowMatching(
        point_clouds=continuous_conditional_data["point_clouds"],
        labels=continuous_conditional_data["labels"],
        config=DefaultConfig(guidance_gamma=2.0, p_uncond=0.1)
    )
    assert not model.discrete_labels

    # Test training
    model.train(training_steps=10, batch_size=4, decay_steps=5)
    assert model.state.step == 10
    assert np.isfinite(model.losses[-1])

    # Test generation
    num_samples = 4
    timesteps = 8
    target_labels = np.random.randn(num_samples, continuous_conditional_data["labels"].shape[1])
    generated_trajectory, weights, final_labels = model.generate_samples(
        num_samples=num_samples,
        timesteps=timesteps,
        generate_labels=target_labels
    )
    assert len(generated_trajectory) == timesteps + 1
    final_samples = generated_trajectory[-1]
    assert final_samples.shape[0] == num_samples
    assert not np.any(np.isnan(final_samples))
    # For continuous labels, the returned labels should be the ones we passed in
    assert np.allclose(final_labels, target_labels)

def test_matched_noise_point_clouds(matched_noise_data):
    """Tests training with a separate, user-provided noise distribution."""
    model = WassersteinFlowMatching(
        point_clouds=matched_noise_data["point_clouds"],
        noise_point_clouds=matched_noise_data["noise_point_clouds"],
        matched_noise=True
    )
    assert model.matched_noise
    assert hasattr(model, 'noise_point_clouds')
    assert model.noise_point_clouds.shape[0] == len(matched_noise_data["noise_point_clouds"])

    # Test training
    model.train(training_steps=10, batch_size=4, decay_steps=5)
    assert model.state.step == 10
    assert np.isfinite(model.losses[-1])

    # Test generation (should use the internal noise function, not the matched one)
    num_samples = 4
    timesteps = 8
    generated_trajectory, weights = model.generate_samples(
        num_samples=num_samples,
        timesteps=timesteps
    )
    assert len(generated_trajectory) == timesteps + 1
    final_samples = generated_trajectory[-1]
    assert final_samples.shape[0] == num_samples
    assert not np.any(np.isnan(final_samples))


@pytest.mark.slow
def test_auto_find_num_iter_runs_without_mocking(unconditional_data):
    """
    This is an integration test that runs the actual auto-finding logic
    for num_sinkhorn_iter. It is marked as 'slow' because it can take
    a significant amount of time to execute.
    
    To run this test specifically:
      pytest -m slow
      
    To skip this test:
      pytest -m "not slow"
    """
    print("\n--- Running SLOW test: test_auto_find_num_iter_runs_without_mocking ---")
    print("This may take a minute or two as it performs real OT calculations...")
    
    # 1. Create a config that will trigger the auto-finding logic
    # Note: This only works for entropic maps where Sinkhorn iterations are relevant.
    config = DefaultConfig(num_sinkhorn_iter=-1, monge_map='entropic')

    # 2. Initialize the model and time the process
    model = WassersteinFlowMatching(
        point_clouds=unconditional_data["point_clouds"],
        config=config
    )


    # 3. Print results for clarity
    found_iter = model.num_sinkhorn_iter
    print(f"Automatically determined num_sinkhorn_iter: {found_iter}")

    # 4. Assert that the process completed and set a valid value
    assert isinstance(found_iter, int)
    assert found_iter > 0
    
    # The returned value must be one of the candidates from the auto_find_num_iter function
    possible_values = [100, 200, 500, 1000, 5000]
    assert found_iter in possible_values, f"Value {found_iter} is not a valid candidate."

    # 5. Assert that the JIT-compiled transport function was configured with the new value
    assert model.transport_plan_jit.keywords['num_iteration'] == found_iter
    print("--- SLOW test completed successfully ---")