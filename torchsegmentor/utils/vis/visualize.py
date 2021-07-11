import cv2
import numpy as np


def colorize(result, color_map=None, normalize=False):
    """Colorize the result by the color map
    
    Arguments:
        result {np array} -- [result]
        color_map {np array or dict} -- [color map])
    
    Returns:
        [np array] -- [colored result]
    """

    if isinstance(color_map, dict):
        unique_label = np.unique(result)
        b = result.copy()
        g = result.copy()
        r = result.copy()
        for cls in unique_label:
            b[b == cls] = color_map[cls][2]
            g[g == cls] = color_map[cls][1]
            r[r == cls] = color_map[cls][0]
        result = np.stack((b, g, r))
    else:
        result = color_map[result]
    if normalize:
        result = result.astype(np.float32)
        result /= 255.0
    return result
