from flax import struct # type: ignore
from typing import Optional

@struct.dataclass
class DefaultConfig:
    
    monge_map: str = 'rounded_matching'
    wasserstein_eps: float = 0.002
    wasserstein_lse: bool = True
    num_sinkhorn_iters: int = 200
    mini_batch_ot_mode: bool = True
    mini_batch_ot_solver: str = 'chamfer'
    minibatch_ot_eps: float = 0.01
    minibatch_ot_lse: bool = True
    noise_type: str = 'chol_normal'
    noise_df_scale: float = 2.0
    scaling: str = 'None'
    scaling_factor: float = 1.0
    guidance_gamma: float = 1.5
    p_uncond: float = 0.1
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 4
    dropout_rate: float = 0.1
    mlp_hidden_dim: int = 512

@struct.dataclass
class SpatialDefaultConfig(DefaultConfig):
    """
    Default configuration for SpatialWormhole, inheriting from DefaultConfig.
    
    Adds parameters specific to handling AnnData objects.
    
    :param rep: (str, optional) The key in `adata.obsm` to use as the expression representation. If None, `adata.X` is used. (default None)
    :param batch_key: (str, optional) The key in `adata.obs` that denotes the sample/batch for each cell. If None, all cells are treated as one batch. (default None)
    """
    noise_type = 'normal'
    rep: Optional[str] = None
    batch_key: Optional[str] = None
    spatial_key: str = 'spatial'
