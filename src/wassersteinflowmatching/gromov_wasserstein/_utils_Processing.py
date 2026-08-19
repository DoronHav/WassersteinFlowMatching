import numpy as np  # type: ignore


def pad_pointclouds(point_clouds: list, weights: list, max_shape: int = -1):
    """Pad point clouds and weights to the same number of points.

    :param point_clouds: List of arrays of shape ``(n_i, d)``.
    :param weights: List of weight vectors of shape ``(n_i,)``.
    :param max_shape: Target number of points; defaults to max over all clouds.
    :return: ``(padded_point_clouds, padded_weights)`` as float32 numpy arrays
             with shapes ``(N, max_shape, d)`` and ``(N, max_shape)``.
    """
    if max_shape == -1:
        max_shape = int(np.max([pc.shape[0] for pc in point_clouds])) + 1
    else:
        max_shape = max_shape + 1

    weights_pad = np.asarray([
        np.concatenate((w, np.zeros(max_shape - pc.shape[0])), axis=0)
        for pc, w in zip(point_clouds, weights)
    ])
    point_clouds_pad = np.asarray([
        np.concatenate(
            [pc, np.zeros([max_shape - pc.shape[0], pc.shape[-1]])], axis=0
        )
        for pc in point_clouds
    ])

    weights_pad = weights_pad / weights_pad.sum(axis=1, keepdims=True)

    return (
        point_clouds_pad[:, :-1].astype("float32"),
        weights_pad[:, :-1].astype("float32"),
    )
