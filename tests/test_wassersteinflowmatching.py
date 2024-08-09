import pytest # type: ignore
import numpy as np # type: ignore

from wassersteinflowmatching import WassersteinFlowMatching

@pytest.fixture
def WassersteinFlowMatchingModel():
    
    point_cloud_sizes = np.random.randint(low = 8, high = 16, size = 64)    
    point_clouds = [np.random.normal(size = [n, 2]) for n in point_cloud_sizes]

    Model = WassersteinFlowMatching(point_clouds = point_clouds)
    return(Model)

def test_train(WassersteinFlowMatchingModel):
    WassersteinFlowMatchingModel.train(training_steps = 1)
    
    
def test_flow(WassersteinFlowMatchingModel):
    WassersteinFlowMatchingModel.train(training_steps = 1)

    WassersteinFlowMatchingModel.generate_samples(num_samples = 10, 
                                                  timesteps = 100)