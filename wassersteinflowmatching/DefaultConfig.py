from flax import struct # type: ignore


@struct.dataclass
class DefaultConfig:
    
    """
    Object with configuration parameters for Wormhole
    
    
    :param dtype: (data type) float point precision for Wormhole model (default jnp.float32)
    :param dist_func_enc: (str) OT metric used for embedding space (default 'S2', could be 'W1', 'S1', 'W2', 'S2', 'GW' and 'GS') 
    :param dist_func_dec: (str) OT metric used for Wormhole decoder loss (default 'S2', could be 'W1', 'S1', 'W2', 'S2', 'GW' and 'GS') 
    :param eps_enc: (float) entropic regularization for embedding OT (default 0.1)
    :param eps_dec: (float) entropic regularization for Wormhole decoder loss (default 0.1)
    :param lse_enc: (bool) whether to use log-sum-exp mode or kernel mode for embedding OT (default False)
    :param lse_dec: (bool) whether to use log-sum-exp mode or kernel mode for decoder OT (default True)
    :param coeff_dec: (float) coefficient for decoder loss (default 1)
    :param scale: (str) how to scale input point clouds ('min_max_total' and scales all point clouds so values are between -1 and 1)
    :param factor: (float) multiplicative factor applied on point cloud coordinates after scaling (default 1)
    :param emb_dim: (int) Wormhole embedding dimention (defulat 128)
    :param num_heads: (int) number of heads in multi-head attention (default 4)
    :param num_layers: (int) number of layers of multi-head attention for Wormhole encoder and decoder (default 3)
    :param qkv_dim: (int) dimention of query, key and value attributes in attention (default 512)
    :param mlp_dim: (int) dimention of hidden layer for fully-connected network after every multi-head attention layer
    :param attention_dropout_rate: (float) dropout rate for attention matrices during training (default 0.1)
    :param kernel_init: (Callable) initializer of kernel weights (default nn.initializers.glorot_uniform())
    :param bias_init: ((Callable) initializer of bias weights (default nn.initializers.zeros_init())
    """ 
    wasserstein_eps: float = 0.001
    wasserstein_lse: bool = True
    mini_batch_ot_mode = True
    minibatch_ot_eps: float = 0.1
    minibatch_ot_lse: bool = False
    scaling: str = 'min_max_total'
    factor: float = 1.0
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 4
    dropout_rate: float = 0.1
    mlp_hidden_dim: int = 1024

