from flax import struct  # type: ignore


@struct.dataclass
class GromovWassersteinFlowMatchingConfig:
    """Default configuration for Gromov-Wasserstein Flow Matching.

    :param gw_epsilon: Entropic regularization for the GW solver.
    :param gw_lse_mode: Use log-sum-exp mode in Sinkhorn (numerically stable).
    :param gw_scale_cost: Divide each intra-geometry cost matrix by its max
        before solving GW, for numerical stability.
    :param teacher_dt: Euler step size for the GW gradient-flow teacher path.
    :param teacher_grad_clip: Clip GW gradient Frobenius norm per step.
    :param rotation_aug: Apply random orthogonal rotations to X0 and X1
        (independently) before computing the teacher path. GW is rotation-
        invariant so the transport plan is unaffected, but the Set Transformer
        sees diverse orientations.
    :param noise_type: Distribution used to sample source noise point clouds.
    :param embedding_dim: Token dimension of the Set Transformer.
    :param num_layers: Number of self-attention encoder blocks.
    :param num_heads: Number of attention heads.
    :param dropout_rate: Dropout probability.
    :param mlp_hidden_dim: Hidden dimension of each feed-forward block.
    """

    # --- GW solver ---
    gw_epsilon: float = 0.05
    gw_lse_mode: bool = True
    gw_scale_cost: bool = True   # divide intra-geometry cost matrices by their max

    # --- Teacher path ---
    teacher_dt: float = 0.05   # 1/teacher_dt gives max teacher steps
    teacher_grad_clip: float = 1.0  # clip GW gradient Frobenius norm to this value
    rotation_aug: bool = False   # random orthogonal rotation augmentation

    # --- Noise distribution ---
    noise_type: str = "uniform"   # 'uniform' | 'normal'

    # --- Architecture ---
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    mlp_hidden_dim: int = 256
