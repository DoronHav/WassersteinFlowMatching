import wassersteinflowmatching.riemannian_wasserstein.utils_Metric as utils_Metric # type: ignore
from wassersteinflowmatching.riemannian_wasserstein.RiemannianWassersteinFlowMatching import ( # type: ignore
    RiemannianWassersteinFlowMatching,
    SourcedRiemannianWassersteinFlowMatching,
)
from wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig # type: ignore


class MetricWassersteinFlowMatching(RiemannianWassersteinFlowMatching):
    """
    Wasserstein Flow Matching using metric geometry (utils_Metric).

    Inherits from RiemannianWassersteinFlowMatching but uses utils_Metric
    for geometry operations. In utils_Metric, geodesics and interpolants
    are approximated numerically, and log maps are computed via gradients
    of the squared distance rather than closed-form expressions.

    :param point_clouds: (list of np.array) list of train-set point clouds to flow match
    :param n_interpolation_steps: (int) number of steps for geodesic interpolation (default 1000)
    :param n_exp_steps: (int) number of steps for exponential map retraction (default: same as n_interpolation_steps)
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized MetricWassersteinFlowMatching model
    """

    _geom_module = utils_Metric

    def __init__(self, point_clouds, n_interpolation_steps=1000, n_exp_steps=None, conditioning=None, config=DefaultConfig, **kwargs):
        self._n_interpolation_steps = n_interpolation_steps
        self._n_exp_steps = n_exp_steps
        super().__init__(point_clouds, conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return getattr(utils_Metric, geom_name)(
            n_interpolation_steps=self._n_interpolation_steps,
            n_exp_steps=self._n_exp_steps,
        )


class SourcedMetricWassersteinFlowMatching(SourcedRiemannianWassersteinFlowMatching):
    """
    Wasserstein Flow Matching with source point clouds using metric geometry (utils_Metric).

    Inherits from SourcedRiemannianWassersteinFlowMatching but uses utils_Metric
    for geometry operations, where geodesics and interpolants are approximated
    numerically and log maps use gradients of the squared distance.

    :param point_clouds: (list of np.array) list of target point clouds
    :param source_point_clouds: (list of np.array) list of source point clouds
    :param n_interpolation_steps: (int) number of steps for geodesic interpolation (default 1000)
    :param n_exp_steps: (int) number of steps for exponential map retraction (default: same as n_interpolation_steps)
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized SourcedMetricWassersteinFlowMatching model
    """

    _geom_module = utils_Metric

    def __init__(self, point_clouds, source_point_clouds, n_interpolation_steps=1000, n_exp_steps=None, matched=False, conditioning=None, config=DefaultConfig, **kwargs):
        self._n_interpolation_steps = n_interpolation_steps
        self._n_exp_steps = n_exp_steps
        super().__init__(point_clouds, source_point_clouds, matched=matched, conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return getattr(utils_Metric, geom_name)(
            n_interpolation_steps=self._n_interpolation_steps,
            n_exp_steps=self._n_exp_steps,
        )
