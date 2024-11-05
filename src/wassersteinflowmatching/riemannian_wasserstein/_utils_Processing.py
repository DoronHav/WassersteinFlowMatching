import numpy as np # type: ignore

def pad_pointclouds(point_clouds, weights, max_shape=-1):
    """
    :meta private:
    """

    if max_shape == -1:
        max_shape = np.max([pc.shape[0] for pc in point_clouds]) + 1
    else:
        max_shape = max_shape + 1


    weights_pad = np.asarray(
        [
            np.concatenate((weight, np.zeros(max_shape - pc.shape[0])), axis=0)
            for pc, weight in zip(point_clouds, weights)
        ]
    )
    point_clouds_pad = np.asarray(
        [
            np.concatenate(
                [pc, np.zeros([max_shape - pc.shape[0], pc.shape[-1]])], axis=0
            )
            for pc in point_clouds
        ]
    )

    weights_pad = weights_pad / weights_pad.sum(axis=1, keepdims=True)

    return (
        point_clouds_pad[:, :-1].astype("float32"),
        weights_pad[:, :-1].astype("float32"),
    )
