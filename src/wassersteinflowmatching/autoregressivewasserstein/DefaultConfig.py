from flax import struct  # type: ignore
from typing import Optional


@struct.dataclass
class ARWFMConfig:

    # OT coupling (cloud-level, before AR selection)
    monge_map: str = 'rounded_matching'  # 'rounded_matching' | 'argmax' | 'entropic' | 'euclidean' | 'random'
    wasserstein_eps: float = 0.002
    wasserstein_lse: bool = True
    num_sinkhorn_iters: Optional[int] = 200

    # Mini-batch OT (assigns noise clouds to target clouds before per-point OT)
    mini_batch_ot_mode: bool = True
    mini_batch_ot_solver: str = 'chamfer'   # 'chamfer' | 'entropic' | 'euclidean' | 'frechet'
    minibatch_ot_eps: float = 0.002
    minibatch_ot_lse: bool = True

    # Noise distribution
    noise_type: str = 'normal'
    noise_df_scale: float = 2.0

    # Context encoder c_phi (permutation-invariant set transformer)
    context_embedding_dim: int = 256
    context_num_layers: int = 4
    context_num_heads: int = 4
    context_dropout_rate: float = 0.0
    context_mlp_hidden_dim: int = 512

    # Flow MLP v_theta (single-point predictor)
    flow_hidden_dim: int = 256
    flow_num_layers: int = 4
    flow_dropout_rate: float = 0.1

    # AR generation order ('random' | 'inside_out' | 'mixture')
    # Used only by SpatialAutoRegressiveWassersteinFM and its subclasses.
    order_mode: str = 'random'
