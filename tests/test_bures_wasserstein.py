import pytest # type: ignore
import numpy as np # type: ignore

from wassersteinflowmatching.bures_wasserstein import BuresWassersteinFlowMatching

@pytest.fixture
def BuresWassersteinFlowMatchingModel():
    
    point_cloud_sizes = np.random.randint(low = 8, high = 16, size = 64)    
    point_clouds = [np.random.normal(size = [n, 2]) for n in point_cloud_sizes]
    means = np.asarray([np.mean(point_cloud, axis = 0) for point_cloud in point_clouds])
    covariances = np.asarray([np.cov(point_cloud, rowvar = False) for point_cloud in point_clouds])
    
    Model = BuresWassersteinFlowMatching(means = means, covariances = covariances)
    return(Model)

def test_train(BuresWassersteinFlowMatchingModel):
    BuresWassersteinFlowMatchingModel.train(training_steps = 1)
    
    
def test_flow(BuresWassersteinFlowMatchingModel):
    BuresWassersteinFlowMatchingModel.train(training_steps = 1)
    BuresWassersteinFlowMatchingModel.generate_samples(num_samples = 10, 
                                                  timesteps = 100)