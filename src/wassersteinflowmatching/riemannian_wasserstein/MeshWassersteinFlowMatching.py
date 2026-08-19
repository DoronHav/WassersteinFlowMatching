import wassersteinflowmatching.riemannian_wasserstein.utils_Mesh as utils_Mesh  # type: ignore
from wassersteinflowmatching.riemannian_wasserstein.RiemannianWassersteinFlowMatching import (  # type: ignore
    RiemannianWassersteinFlowMatching,
    SourcedRiemannianWassersteinFlowMatching,
)
from wassersteinflowmatching.riemannian_wasserstein.DefaultConfig import DefaultConfig  # type: ignore


class MeshWassersteinFlowMatching(RiemannianWassersteinFlowMatching):
    """
    Wasserstein Flow Matching on a general triangular-mesh manifold (e.g. the Stanford bunny).

    Implements the general-geometry case of Riemannian Flow Matching (Chen & Lipman, ICLR 2024,
    Section 3.3) using a spectral (biharmonic) premetric computed from the mesh's Laplace-Beltrami
    eigenfunctions. Geometry operations are provided by ``utils_Mesh.TriangleMesh``, which derives
    log/exp maps and interpolants numerically from the spectral squared distance.

    :param point_clouds: (list of np.array) train-set point clouds living on the mesh surface
    :param mesh: path to a .obj/.ply file, or a ``(V, F)`` tuple of vertices and faces
    :param k: (int) number of eigenfunctions for the spectral distance (default 200)
    :param spectral: (str) 'biharmonic' (default, tuning-free) or 'diffusion'
    :param tau: (float) diffusion-distance time parameter (only if spectral='diffusion')
    :param normalize: (bool) rescale the mesh into [-0.99, 0.99]^3 (default True)
    :param n_interpolation_steps: (int) substeps for the marching geodesic interpolant (default 50)
    :param n_exp_steps: (int) substeps for the retraction exp map (default: n_interpolation_steps)
    :param cpu_projection: (bool) project the input point clouds onto the mesh on CPU (numpy) if
        True, or on GPU (jax) if False (default True). GPU is faster for large meshes/clouds.
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized MeshWassersteinFlowMatching model
    """

    _geom_module = utils_Mesh

    def __init__(self, point_clouds, mesh, k=200, spectral="biharmonic", tau=1.0,
                 normalize=True, n_interpolation_steps=50, n_exp_steps=None,
                 cpu_projection=True, conditioning=None, config=DefaultConfig, **kwargs):
        self._mesh = mesh
        self._k = k
        self._spectral = spectral
        self._tau = tau
        self._normalize = normalize
        self._n_interpolation_steps = n_interpolation_steps
        self._n_exp_steps = n_exp_steps
        kwargs.setdefault("geom", "TriangleMesh")
        kwargs.setdefault("cpu_projection", cpu_projection)
        super().__init__(point_clouds, conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return utils_Mesh.TriangleMesh(
            mesh=self._mesh,
            k=self._k,
            spectral=self._spectral,
            tau=self._tau,
            normalize=self._normalize,
            n_interpolation_steps=self._n_interpolation_steps,
            n_exp_steps=self._n_exp_steps,
        )


class SourcedMeshWassersteinFlowMatching(SourcedRiemannianWassersteinFlowMatching):
    """
    Sourced Wasserstein Flow Matching on a triangular-mesh manifold.

    Like :class:`MeshWassersteinFlowMatching` but transports between explicit source and target
    point-cloud distributions on the mesh (rather than from a base noise distribution).

    :param point_clouds: (list of np.array) target point clouds on the mesh
    :param source_point_clouds: (list of np.array) source point clouds on the mesh
    :param mesh: path to a .obj/.ply file, or a ``(V, F)`` tuple of vertices and faces
    :param k: (int) number of eigenfunctions for the spectral distance (default 200)
    :param spectral: (str) 'biharmonic' (default) or 'diffusion'
    :param tau: (float) diffusion-distance time parameter (only if spectral='diffusion')
    :param normalize: (bool) rescale the mesh into [-0.99, 0.99]^3 (default True)
    :param n_interpolation_steps: (int) substeps for the marching interpolant (default 50)
    :param n_exp_steps: (int) substeps for the retraction exp map (default: n_interpolation_steps)
    :param cpu_projection: (bool) project the input point clouds onto the mesh on CPU (numpy) if
        True, or on GPU (jax) if False (default True). GPU is faster for large meshes/clouds.

    :return: initialized SourcedMeshWassersteinFlowMatching model
    """

    _geom_module = utils_Mesh

    def __init__(self, point_clouds, source_point_clouds, mesh, k=200, spectral="biharmonic",
                 tau=1.0, normalize=True, n_interpolation_steps=50, n_exp_steps=None,
                 cpu_projection=True, matched=False, conditioning=None, config=DefaultConfig, **kwargs):
        self._mesh = mesh
        self._k = k
        self._spectral = spectral
        self._tau = tau
        self._normalize = normalize
        self._n_interpolation_steps = n_interpolation_steps
        self._n_exp_steps = n_exp_steps
        kwargs.setdefault("geom", "TriangleMesh")
        kwargs.setdefault("cpu_projection", cpu_projection)
        super().__init__(point_clouds, source_point_clouds, matched=matched,
                         conditioning=conditioning, config=config, **kwargs)

    def _create_geom_utils(self, geom_name):
        return utils_Mesh.TriangleMesh(
            mesh=self._mesh,
            k=self._k,
            spectral=self._spectral,
            tau=self._tau,
            normalize=self._normalize,
            n_interpolation_steps=self._n_interpolation_steps,
            n_exp_steps=self._n_exp_steps,
        )
