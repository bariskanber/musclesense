import numpy
import keras
import keras.backend as K
from keras.losses import binary_crossentropy

# credit: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# credit: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1) # is abs needed?
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_proper(y_true, y_pred, smooth=1e-6):
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def DiceLoss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def DiceBCELoss(targets, inputs, smooth=1e-6):    
     
    #flatten label and prediction tensors
    #inputs = K.flatten(inputs)
    #targets = K.flatten(targets)
    
    #intersection = K.sum(K.dot(targets, inputs))    
    #dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = DiceLoss(targets, inputs) + binary_crossentropy(targets, inputs)
    
    return Dice_BCE
