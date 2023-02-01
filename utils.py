import numpy as np
from scipy import ndimage as nd

def numpy_dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def scale2D(inmat,targetsize_y,targetsize_x,order=3,mode='constant',cval=0.0,prefilter=True):
    z=[float(targetsize_y)/inmat.shape[0],float(targetsize_x)/inmat.shape[1]]

    retmat = nd.interpolation.zoom(inmat,zoom=z,order=order,mode=mode,cval=cval,prefilter=prefilter)

    if retmat.shape!=(targetsize_y,targetsize_x):
        print('scale2D did not return the expected size output',retmat.shape)
        raise Exception('scale2D error')

    return retmat

def scale3D(inmat,targetsize_z,targetsize_y,targetsize_x,order=3,mode='constant',cval=0.0,prefilter=True):
    z=[float(targetsize_z)/inmat.shape[0],float(targetsize_y)/inmat.shape[1],float(targetsize_x)/inmat.shape[2]]

    retmat = nd.interpolation.zoom(inmat,zoom=z,order=order,mode=mode,cval=cval,prefilter=prefilter)

    if retmat.shape!=(targetsize_z,targetsize_y,targetsize_x):
        print('scale3D did not return the expected size output',retmat.shape)
        raise Exception('scale3D error')

    return retmat

