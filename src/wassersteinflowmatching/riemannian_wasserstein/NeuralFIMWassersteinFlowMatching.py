import wassersteinflowmatching.riemannian_wasserstein.utils_NeuralFIM as utils_NeuralFIM  # type: ignore
from wassersteinflowmatching.riemannian_wasserstein.RiemannianWassersteinFlowMatching import (  # type: ignore
    RiemannianWassersteinFlowMatching,
    SourcedRiemannianWassersteinFlowMatching,
)
from wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig  # type: ignore


class NeuralFIMWassersteinFlowMatching(RiemannianWassersteinFlowMatching):
    """
    Wasserstein Flow Matching on a learned Neural-FIM Fisher-Rao geometry.

    The geometry is a Fisher-Rao sphere metric pulled back through a pretrained (non-invertible)
    softmax encoder f_psi (see ``utils_NeuralFIM.train_neural_fim``, Fasina et al. ICML 2023).
    Because f_psi is not invertible, log/exp/interpolant are derived via autodiff + multi-step
    retraction (``utils_Metric.generic_riemannian``), unlike ``PullbackFlowWassersteinFlowMatching``,
    which has closed-form primitives. Calibrate ``n_interpolation_steps`` for cost before a full run.

    :param point_clouds: (list of np.array) train-set point clouds in the encoder's ambient space
    :param fim_net: pretrained ``utils_NeuralFIM.NeuralFIMEncoder`` (architecture only, untrained)
    :param fim_params: pretrained encoder parameters (pytree matching ``fim_net``)
    :param n_interpolation_steps: (int) marching-scan substeps (default 1000)
    :param n_exp_steps: (int) substeps for the retraction exp map (default: same as above)
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized NeuralFIMWassersteinFlowMatching model
    """

    _geom_module = utils_NeuralFIM

    def __init__(self, point_clouds, fim_net, fim_params, n_interpolation_steps=1000,
                n_exp_steps=None, conditioning=None, config=DefaultConfig, **kwargs):
        self._fim_net = fim_net
        self._fim_params = fim_params
        self._n_interpolation_steps = n_interpolation_steps
        self._n_exp_steps = n_exp_steps
        kwargs.setdefault("geom", "NeuralFIM")
        kwargs.setdefault("cpu_projection", False)
        super().__init__(point_clouds, conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return utils_NeuralFIM.NeuralFIM(
            self._fim_net, self._fim_params,
            n_interpolation_steps=self._n_interpolation_steps,
            n_exp_steps=self._n_exp_steps,
        )


class SourcedNeuralFIMWassersteinFlowMatching(SourcedRiemannianWassersteinFlowMatching):
    """
    Sourced Wasserstein Flow Matching on a learned Neural-FIM Fisher-Rao geometry.

    Like :class:`NeuralFIMWassersteinFlowMatching` but transports between explicit source and
    target point-cloud distributions rather than from a base noise distribution.

    :param point_clouds: (list of np.array) target point clouds in the encoder's ambient space
    :param source_point_clouds: (list of np.array) source point clouds in the encoder's ambient space
    :param fim_net: pretrained ``utils_NeuralFIM.NeuralFIMEncoder`` (architecture only, untrained)
    :param fim_params: pretrained encoder parameters (pytree matching ``fim_net``)
    :param n_interpolation_steps: (int) marching-scan substeps (default 1000)
    :param n_exp_steps: (int) substeps for the retraction exp map (default: same as above)

    :return: initialized SourcedNeuralFIMWassersteinFlowMatching model
    """

    _geom_module = utils_NeuralFIM

    def __init__(self, point_clouds, source_point_clouds, fim_net, fim_params,
                n_interpolation_steps=1000, n_exp_steps=None, matched=False, conditioning=None,
                config=DefaultConfig, **kwargs):
        self._fim_net = fim_net
        self._fim_params = fim_params
        self._n_interpolation_steps = n_interpolation_steps
        self._n_exp_steps = n_exp_steps
        kwargs.setdefault("geom", "NeuralFIM")
        kwargs.setdefault("cpu_projection", False)
        super().__init__(point_clouds, source_point_clouds, matched=matched,
                         conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return utils_NeuralFIM.NeuralFIM(
            self._fim_net, self._fim_params,
            n_interpolation_steps=self._n_interpolation_steps,
            n_exp_steps=self._n_exp_steps,
        )
