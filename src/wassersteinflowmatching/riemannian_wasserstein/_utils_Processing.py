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

def pad_pointclouds_features_and_masks(point_clouds, weights, masks, max_shape=-1):
    """
    Pads point clouds in both number of points (axis 0) and feature dimension (axis 1).
    """

    if max_shape == -1:
        max_shape = np.max([pc.shape[0] for pc in point_clouds]) + 1
    else:
        max_shape = max_shape + 1
    
    max_dim = np.max([pc.shape[1] for pc in point_clouds])
    max_residues = max_dim // 7

    weights_pad = np.asarray(
        [
            np.concatenate((weight, np.zeros(max_shape - pc.shape[0])), axis=0)
            for pc, weight in zip(point_clouds, weights)
        ]
    )
    
    point_clouds_pad = []
    for pc in point_clouds:
        # Pad points (axis 0)
        pad_n = max_shape - pc.shape[0]
        pc_pad_n = np.concatenate([pc, np.zeros([pad_n, pc.shape[1]])], axis=0)
        
        # Pad features (axis 1)
        pad_c = max_dim - pc.shape[1]
        pc_pad_c = np.concatenate([pc_pad_n, np.zeros([max_shape, pad_c])], axis=1)
        
        point_clouds_pad.append(pc_pad_c)
    point_clouds_pad = np.asarray(point_clouds_pad)
    
    masks_pad = []
    for mask in masks:
        # Pad points (axis 0)
        pad_n = max_shape - mask.shape[0]
        mask_pad_n = np.concatenate([mask, np.zeros([pad_n, mask.shape[1]])], axis=0)
        
        # Pad features (axis 1) - mask dimension is residues = dim // 7
        pad_r = max_residues - mask.shape[1]
        mask_pad_r = np.concatenate([mask_pad_n, np.zeros([max_shape, pad_r])], axis=1)
        
        masks_pad.append(mask_pad_r)
    masks_pad = np.asarray(masks_pad)

    weights_pad = weights_pad / weights_pad.sum(axis=1, keepdims=True)

    return (
        point_clouds_pad[:, :-1].astype("float32"),
        weights_pad[:, :-1].astype("float32"),
        masks_pad[:, :-1].astype("float32"),
    )
