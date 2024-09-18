import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as nd

def numpy_dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def scale2D(inmat, targetsize_y, targetsize_x, order=3, mode='constant', cval=0.0, prefilter=True):
    z = [float(targetsize_y)/inmat.shape[0], float(targetsize_x)/inmat.shape[1]]

    retmat = nd.interpolation.zoom(inmat, zoom=z, order=order, mode=mode, cval=cval, prefilter=prefilter)

    if retmat.shape != (targetsize_y, targetsize_x):
        print('scale2D did not return the expected size output', retmat.shape)
        raise Exception('scale2D error')

    return retmat


def scale3D(inmat, targetsize_z, targetsize_y, targetsize_x, order=3, mode='constant', cval=0.0, prefilter=True):
    z = [float(targetsize_z)/inmat.shape[0], float(targetsize_y)/inmat.shape[1], float(targetsize_x)/inmat.shape[2]]

    retmat = nd.interpolation.zoom(inmat, zoom=z, order=order, mode=mode, cval=cval, prefilter=prefilter)

    if retmat.shape != (targetsize_z, targetsize_y, targetsize_x):
        print('scale3D did not return the expected size output', retmat.shape)
        raise Exception('scale3D error')

    return retmat

def checkDixonImage(dixon_img) -> bool:
    hist, _ = np.histogram(dixon_img, bins=20)
    if np.argmax(hist) == 0: return True
    else:
        return False
        
def get_fmf(pattern: str, case_sensitive = False) -> str:
    """Get first file matching given pattern(s) or raise an exception

    Args:
        pattern (str/list): File pattern(s) to match

    Returns:
        filename: The first matching filename
        
    Example:
        get_fmf('/home/myfile.*')
    """
    if type(pattern) is str: pattern = [pattern]
    
    for p in pattern:
        if (sys.version_info.major==3 and sys.version_info.minor>=12) or sys.version_info.major>3:
            files = [*Path.cwd().glob(p, case_sensitive=case_sensitive)]
        else:
            print('WARNING: Python version is < 3.12')
            files = [*Path.cwd().glob(p)]

        if len(files) > 0:
            return str(files[0])

    raise Exception(f'ERROR: No files found matching {pattern}')
