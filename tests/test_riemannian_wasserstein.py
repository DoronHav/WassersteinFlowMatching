import pytest # type: ignore
import numpy as np # type: ignore

from src.wassersteinflowmatching.riemannian_wasserstein import RiemannianWassersteinFlowMatching

@pytest.fixture
def RiemannianWassersteinFlowMatchingModel():
    
    point_cloud_sizes = np.random.randint(low = 8, high = 16, size = 64)    
    point_clouds = [np.random.normal(size = [n, 2]) for n in point_cloud_sizes]

    Model = RiemannianWassersteinFlowMatching(point_clouds = point_clouds)
    return(Model)

def test_train(RiemannianWassersteinFlowMatchingModel):
    RiemannianWassersteinFlowMatchingModel.train(training_steps = 1)
    
    
def test_flow(RiemannianWassersteinFlowMatchingModel):
    RiemannianWassersteinFlowMatchingModel.train(training_steps = 1)
    RiemannianWassersteinFlowMatchingModel.generate_samples(num_samples = 10, 
                                                  timesteps = 100)