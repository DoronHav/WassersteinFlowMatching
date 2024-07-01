import pytest

import anndata
import numpy as np

from wassersteinwormhole import Wormhole

@pytest.fixture
def WormholeModel():
    
    point_cloud_sizes_train = np.random.randint(low = 8, high = 16, size = 64)
    point_cloud_sizes_test = np.random.randint(low = 8, high = 16, size = 32)
    
    pc_train = [np.random.normal(size = [n, 2]) for n in point_cloud_sizes_train]
    pc_test = [np.random.normal(size = [n, 2]) for n in point_cloud_sizes_test]
    
    weights_train = [np.random.uniform(low = 0, high = 1, size = n) for n in point_cloud_sizes_train]
    weights_test = [np.random.uniform(low = 0, high = 1, size = n) for n in point_cloud_sizes_test]
    
    Model = Wormhole(point_clouds = pc_train, weights = weights_train, point_clouds_test = pc_test, weights_test = weights_test)
    return(Model)

def test_train(WormholeModel):
    WormholeModel.train(training_steps = 1)
    
    
def test_encode(WormholeModel):
    WormholeModel.train(training_steps = 1)
    
    train_encodings = WormholeModel.encode(WormholeModel.point_clouds, WormholeModel.weights)
    test_encodings = WormholeModel.encode(WormholeModel.point_clouds_test, WormholeModel.weights_test)
    
    assert train_encodings.shape[0] == WormholeModel.point_clouds.shape[0]
    assert test_encodings.shape[0] == WormholeModel.point_clouds_test.shape[0]
    
    assert train_encodings.shape[1] == WormholeModel.config.emb_dim
    assert test_encodings.shape[1] == WormholeModel.config.emb_dim
    
def test_decoder(WormholeModel):
    WormholeModel.train(training_steps = 1)
    
    train_encodings = WormholeModel.encode(WormholeModel.point_clouds, WormholeModel.weights)
    test_encodings = WormholeModel.encode(WormholeModel.point_clouds_test, WormholeModel.weights_test)
    
    train_decodings = WormholeModel.decode(train_encodings)
    test_decodings = WormholeModel.decode(test_encodings)

    assert train_decodings.shape[0] == WormholeModel.point_clouds.shape[0]
    assert test_decodings.shape[0] == WormholeModel.point_clouds_test.shape[0]
    
    assert train_decodings.shape[1] == WormholeModel.out_seq_len
    assert test_decodings.shape[1] == WormholeModel.out_seq_len