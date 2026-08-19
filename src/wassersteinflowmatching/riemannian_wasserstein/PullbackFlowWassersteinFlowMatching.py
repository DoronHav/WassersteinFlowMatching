import wassersteinflowmatching.riemannian_wasserstein.utils_PullbackFlow as utils_PullbackFlow  # type: ignore
from wassersteinflowmatching.riemannian_wasserstein.RiemannianWassersteinFlowMatching import (  # type: ignore
    RiemannianWassersteinFlowMatching,
    SourcedRiemannianWassersteinFlowMatching,
)
from wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig  # type: ignore


class PullbackFlowWassersteinFlowMatching(RiemannianWassersteinFlowMatching):
    """
    Wasserstein Flow Matching on a learned pullback-flow geometry.

    The geometry is a Riemannian metric on ambient space, pulled back from a flat latent space
    through a pretrained invertible flow (see ``utils_PullbackFlow.train_pullback_flow``, which
    fits the flow so that latent-space Euclidean distance approximates a data-driven diffusion
    geometry). Because the flow is invertible, every geometry primitive (distance, log/exp map,
    geodesic interpolant, tangent norm) is closed-form -- unlike ``MeshWassersteinFlowMatching``,
    which needs a multi-step marching-scan retraction because its premetric has no known global
    isometry to a flat space.

    :param point_clouds: (list of np.array) train-set point clouds in the flow's ambient space
    :param flow_net: pretrained ``utils_PullbackFlow.PullbackFlowNet`` (architecture only, untrained)
    :param flow_params: pretrained flow parameters (pytree matching ``flow_net``)
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized PullbackFlowWassersteinFlowMatching model
    """

    _geom_module = utils_PullbackFlow

    def __init__(self, point_clouds, flow_net, flow_params, conditioning=None,
                config=DefaultConfig, **kwargs):
        self._flow_net = flow_net
        self._flow_params = flow_params
        kwargs.setdefault("geom", "PullbackFlow")
        kwargs.setdefault("cpu_projection", False)
        super().__init__(point_clouds, conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return utils_PullbackFlow.PullbackFlow(self._flow_net, self._flow_params)


class SourcedPullbackFlowWassersteinFlowMatching(SourcedRiemannianWassersteinFlowMatching):
    """
    Sourced Wasserstein Flow Matching on a learned pullback-flow geometry.

    Like :class:`PullbackFlowWassersteinFlowMatching` but transports between explicit source and
    target point-cloud distributions rather than from a base noise distribution.

    :param point_clouds: (list of np.array) target point clouds in the flow's ambient space
    :param source_point_clouds: (list of np.array) source point clouds in the flow's ambient space
    :param flow_net: pretrained ``utils_PullbackFlow.PullbackFlowNet`` (architecture only, untrained)
    :param flow_params: pretrained flow parameters (pytree matching ``flow_net``)

    :return: initialized SourcedPullbackFlowWassersteinFlowMatching model
    """

    _geom_module = utils_PullbackFlow

    def __init__(self, point_clouds, source_point_clouds, flow_net, flow_params, matched=False,
                conditioning=None, config=DefaultConfig, **kwargs):
        self._flow_net = flow_net
        self._flow_params = flow_params
        kwargs.setdefault("geom", "PullbackFlow")
        kwargs.setdefault("cpu_projection", False)
        super().__init__(point_clouds, source_point_clouds, matched=matched,
                         conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return utils_PullbackFlow.PullbackFlow(self._flow_net, self._flow_params)
