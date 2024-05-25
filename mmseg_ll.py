#!miniconda3/bin/python
import os
import numpy as np
import nibabel as nib
import sys
import glob
import time
import random
import multiprocessing
from joblib import Parallel, delayed
import argparse
import urllib.request
from tqdm import tqdm
import traceback
from sklearn.model_selection import GroupKFold

import pynvml
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
from torchvision.transforms.functional import resized_crop
import segmentation_models_pytorch as smp

from convertMRCCentreMaskToBinary import convertMRCCentreMaskToBinary
from convertMRCCentreMaskToStandard import convertMRCCentreMaskToStandard
from mmseg_utils import numpy_dice_coefficient, scale2D, checkDixonImage
from nifti_tools import save_nifti

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

DEBUG = True
llshortdict = {'thigh': 'th', 'calf': 'cf'}

scale_down_factor = 1

# calf: 512x240; thigh: 512x256
target_size_y, target_size_x = 320//scale_down_factor, 160//scale_down_factor

RUNTIME_PARAMS = {'smoketest': False, 'caution': False, 'batch_size': 4, 'lr': 1E-3, 'patience': 5}

MODULE_NAME = os.path.basename(__file__)
INSTALL_DIR = os.path.dirname(os.path.realpath(__file__))
print('INSTALL_DIR', INSTALL_DIR)
print('MODULE_NAME', MODULE_NAME)

modalities_dixon_345_460_575 = 'dixon_345_460_575'
modalities_t1 = 't1'
modalities_t2_stir = 't2_stir'

available_modalities = [modalities_dixon_345_460_575, modalities_t1, modalities_t2_stir]

def get_subject_id_from_DIR(DIR):
    DIR = DIR.split('^')  # e.g. thigh^data/brcalskd/BRCALSKD_056C
    assert (len(DIR) >= 2)
    DIR = DIR[1]

    if 'arimoclomol' in DIR: 
        subject = os.path.basename(DIR) # e.g. kansas_11_011_12m
        if subject.startswith('kansas'):
            assert (len(subject) >= 13)
            subject = 'arimoclomol.'+subject[:13]
            print(subject)
            return subject
        elif subject.startswith('nhnn'): # e.g. nhnn_020-031_20m
            assert (len(subject) >= 12)
            subject = 'arimoclomol.'+subject[:12]
            print(subject)
            return subject
        else:
            assert(False)
                
    if 'dhmn' in DIR: 
        subject = os.path.basename(DIR) # e.g. gait_110b
        assert (len(subject) >= 8)
        subject = 'dhmn.'+subject[:8]
        #print(subject)
        return subject
        
    if 'mdacmt' in DIR: 
        subject = os.path.basename(DIR) # e.g. iowa_066a
        assert (len(subject) == 9)
        subject = 'mdacmt.'+subject[:8]
        #print(subject)
        return subject
        
    if 'alscs' in DIR: 
        subject = os.path.basename(DIR).replace('Baloh_', '') # e.g. Baloh_42350_001B
        assert (len(subject) >= 9)
        subject = 'alscs.'+subject[:9]
        #print(subject)
        return subject
        
    if 'BRCALSKD' in DIR:
        subject = os.path.basename(DIR).replace('BRCALSKD_', '')
        assert (len(subject) == 4)
        return 'BRCALSKD.'+subject[:3]

    if 'HYPOPP' in DIR.upper():
        subject = os.path.basename(DIR)
        assert (len(subject) == 5)
        assert (subject[3] == '_')
        return 'HYPOPP.'+subject[:3]

    if 'ibmcmt_p1' in DIR:
        subject = os.path.basename(DIR).replace('p1-', '')
        assert (len(subject) == 4)
        return 'ibmcmt_p1.'+subject[:3]

    if 'ibmcmt_p2' in DIR:
        subject = os.path.basename(DIR).replace('p2-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p3' in DIR:
        subject = os.path.basename(DIR).replace('p3-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p4' in DIR:
        subject = os.path.basename(DIR).replace('p4-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p5' in DIR:
        subject = os.path.basename(DIR).replace('p5-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    if 'ibmcmt_p6' in DIR:
        subject = os.path.basename(DIR).replace('p6-', '')
        assert (len(subject) == 4 or len(subject) == 3)
        return 'ibmcmt_p2-6.'+subject[:3]

    raise Exception('Unable to determine subject ID')


MASK_VALIDITY_VALID = 1
MASK_VALIDITY_BLANKMASK = 2
MASK_VALIDITY_BAD = 3
MASK_VALIDITY_SINGLESIDED = 4

def valid_mask(mask_original, help_str):
    validity = __valid_mask(mask_original, help_str)
    
    if DEBUG and '^side' in help_str:
        plt.figure()
        plt.imshow(mask_original)
        plt.title(help_str+', validity = '+str(validity))
        if not os.path.exists('__DEBUG'): os.mkdir('__DEBUG')
        plt.savefig('__DEBUG/'+help_str.replace('/','_')+'.png')
        plt.close()

    return validity
    
def __valid_mask(mask_original, help_str):
    mask = mask_original.copy()
    if RUNTIME_PARAMS['multiclass']:
        if RUNTIME_PARAMS['al'] == 'calf':
            mask[mask_original > 0] = 1
            mask[mask_original == 7] = 0  # right tibia marrow
            mask[mask_original == 17] = 0  # left tibia marrow
        else:
            mask[mask_original > 0] = 1
            mask[mask_original == 11] = 0  # right femur marrow
            mask[mask_original == 31] = 0  # left femur marrow

    if not np.array_equal(mask.shape, [target_size_y, target_size_x]):
        print('np.array_equal(mask.shape,[target_size_y,target_size_x]) is false',
              mask.shape, [target_size_y, target_size_x])
        assert (False)

    mask_sum = np.sum(mask)
    if (mask_sum == 0):
        return MASK_VALIDITY_BLANKMASK

    if not np.array_equal(np.unique(mask), [0, 1]):
        raise Exception("Mask values not 0 and 1: "+help_str)

    if mask_sum/np.prod(mask.shape) < 0.01:
        if DEBUG:
            print('WARNING: %s with a value of %f, assuming this is not a valid mask' %
                  (help_str, mask_sum/np.prod(mask.shape)))
        return MASK_VALIDITY_BAD

    QQ = np.where(mask == 1)
    diffY = np.max(QQ[0])-np.min(QQ[0])
    assert diffY > 0, 'diffY needs to be >0'
    ratio = float(diffY)/mask.shape[0]
    if ratio < 0.5:
        if DEBUG:
            print('WARNING: ratio (%f)<0.5 for %s, assuming this is a one sided mask' % (ratio,help_str))
        return MASK_VALIDITY_SINGLESIDED

    return MASK_VALIDITY_VALID

def load_BFC_image(filename, test):
    if True:
        if not os.path.exists(filename):
            raise Exception(f'ERROR: the following file does not exist {filename}')
        nibobj = nib.load(filename)
        return nibobj, nibobj.get_fdata()

def get_fmf(pattern):
    """Get first file matching given pattern or raise an exception

    Args:
        pattern (str): File pattern to match

    Returns:
        filename: The first matching filename
        
    Example:
        get_fmf('/home/myfile.*')
    """
    files = glob.glob(pattern)
    if len(files) < 1:
        print(f'ERROR: No files found matching {pattern}')
        assert (False)
    return files[0]

def load_case_base(inputdir, DIR, multiclass, test):
    CAUTION=False
    if DEBUG:
        print('load_case', DIR)
    TK = DIR.split('^')
    assert (len(TK) >= 2)
    ll = TK[0]
    DIR = TK[1]

    if RUNTIME_PARAMS['modalities']==modalities_t1:
        if inputdir != 'train' and inputdir != 'validate':
            filename = get_fmf(os.path.join(DIR, 't1.nii*'))
        else:
            filename = get_fmf(os.path.join(DIR, 'nii/t1w_%s_*.nii*'%(ll)))
    elif RUNTIME_PARAMS['modalities']==modalities_t2_stir:
        if inputdir != 'train' and inputdir != 'validate':
            filename = get_fmf(os.path.join(DIR, 't2_stir.nii*'))
        else:
            filename = get_fmf(os.path.join(DIR, 'nii/stir_%s_*.nii*'%(ll)))
    elif inputdir != 'train' and inputdir != 'validate':
        filename = get_fmf(os.path.join(DIR, 'dixon345.nii*'))
    else:
        filestorename = glob.glob(os.path.join(DIR, 'nii/*-dix3d_TE*.nii.gz'))
        for filetorename in filestorename:
            print('Renaming '+filetorename)
            os.rename(filetorename, filetorename.replace('-dix3d_','-Dixon_'))
            
        dixfile = glob.glob(os.path.join(DIR, 'nii/*-Dixon*TE*3*45_'+ll+'.nii.gz'))
        if len(dixfile) == 0:
            dixfile = glob.glob(os.path.join(DIR, 'nii/*-Dixon*TE*3*45_'+llshortdict[ll]+'.nii.gz'))
        if len(dixfile) == 0:
            dixfile = glob.glob(os.path.join(DIR, 'nii/*DIXON*TE*3*45_'+llshortdict[ll].upper()+'*.nii.gz'))
        assert len(dixfile) == 2, 'Failed len(dixfile)==2 for '+DIR

        id1 = os.path.basename(dixfile[0]).split('-Dixon')[0].replace('-', '.')
        id2 = os.path.basename(dixfile[1]).split('-Dixon')[0].replace('-', '.')
        try:
            id1 = float(id1)
            id2 = float(id2)
        except:
            id1 = str(id1)
            id2 = str(id2)
        assert (id1 != id2)

        if id1 > id2:
            dixfile = dixfile[::-1]
            
        filename = dixfile[0]
        if False:
            from register_t1_t2_stir_to_dixon import register_t1_t2_stir_to_dixon
            register_t1_t2_stir_to_dixon(filename, ll, llshortdict)

    dixon_345imgobj, dixon_345img = load_BFC_image(filename, test)
    if RUNTIME_PARAMS['modalities']==modalities_dixon_345_460_575:
        assert checkDixonImage(dixon_345img), filename+' may be a phase image'

    if RUNTIME_PARAMS['modalities']==modalities_t1:
        if inputdir != 'train' and inputdir != 'validate':
            filename = get_fmf(os.path.join(DIR, 't1.nii*'))
        else:
            filename = get_fmf(os.path.join(DIR, 'nii/t1w_%s_*.nii*'%(ll)))
    elif RUNTIME_PARAMS['modalities']==modalities_t2_stir:
        if inputdir != 'train' and inputdir != 'validate':
            filename = get_fmf(os.path.join(DIR, 't2_stir.nii*'))
        else:
            filename = get_fmf(os.path.join(DIR, 'nii/stir_%s_*.nii*'%(ll)))
    elif inputdir != 'train' and inputdir != 'validate':
        filename = get_fmf(os.path.join(DIR, 'dixon460.nii*'))
    else:
        dixfile = glob.glob(os.path.join(DIR, 'nii/*-Dixon*TE*4*60_'+ll+'.nii.gz'))
        if len(dixfile) == 0:
            dixfile = glob.glob(os.path.join(DIR, 'nii/*-Dixon*TE*4*60_'+llshortdict[ll]+'.nii.gz'))
        if len(dixfile) == 0:
            dixfile = glob.glob(os.path.join(DIR, 'nii/*DIXON*TE*4*60_'+llshortdict[ll].upper()+'*.nii.gz'))
        assert len(dixfile) == 2, 'Failed len(dixfile)==2 for '+DIR

        id1 = os.path.basename(dixfile[0]).split('-Dixon')[0].replace('-', '.')
        id2 = os.path.basename(dixfile[1]).split('-Dixon')[0].replace('-', '.')
        try:
            id1 = float(id1)
            id2 = float(id2)
        except:
            id1 = str(id1)
            id2 = str(id2)
        assert (id1 != id2)

        if id1 > id2:
            dixfile = dixfile[::-1]
            
        filename = dixfile[0]

    dixon_460imgobj, dixon_460img = load_BFC_image(filename, test)
    if not np.array_equal(dixon_345imgobj.header.get_zooms(),dixon_460imgobj.header.get_zooms()):
       CAUTION=True
       if DEBUG: print('CAUTION: dixon_345 and dixon_460 image resolutions are different for '+DIR)

    if RUNTIME_PARAMS['modalities']==modalities_dixon_345_460_575:
        if 0 and filename.replace('data/','') == 'ibmcmt_p1/p1-010a/nii/0037-Dixon_TE_460_cf.nii.gz':
            pass
        else:
            assert checkDixonImage(dixon_460img), filename+' may be a phase image'

    if RUNTIME_PARAMS['modalities']==modalities_t1:
        if inputdir != 'train' and inputdir != 'validate':
            filename = get_fmf(os.path.join(DIR, 't1.nii*'))
        else:
            filename = get_fmf(os.path.join(DIR, 'nii/t1w_%s_*.nii*'%(ll)))
    elif RUNTIME_PARAMS['modalities']==modalities_t2_stir:
        if inputdir != 'train' and inputdir != 'validate':
            filename = get_fmf(os.path.join(DIR, 't2_stir.nii*'))
        else:
            filename = get_fmf(os.path.join(DIR, 'nii/stir_%s_*.nii*'%(ll)))
    elif inputdir != 'train' and inputdir != 'validate':
        filename = get_fmf(os.path.join(DIR, 'dixon575.nii*'))
    else:
        dixfile = glob.glob(os.path.join(DIR, 'nii/*-Dixon*TE*5*75_'+ll+'.nii.gz'))
        if len(dixfile) == 0:
            dixfile = glob.glob(os.path.join(DIR, 'nii/*-Dixon*TE*5*75_'+llshortdict[ll]+'.nii.gz'))
        if len(dixfile) == 0:
            dixfile = glob.glob(os.path.join(DIR, 'nii/*DIXON*TE*5*75_'+llshortdict[ll].upper()+'*.nii.gz'))
        assert len(dixfile) == 2, 'Failed len(dixfile)==2 for '+DIR

        id1 = os.path.basename(dixfile[0]).split('-Dixon')[0].replace('-', '.')
        id2 = os.path.basename(dixfile[1]).split('-Dixon')[0].replace('-', '.')
        try:
            id1 = float(id1)
            id2 = float(id2)
        except:
            id1 = str(id1)
            id2 = str(id2)
        assert (id1 != id2)

        if id1 > id2:
            dixfile = dixfile[::-1]
            
        filename = dixfile[0]

    dixon_575imgobj, dixon_575img = load_BFC_image(filename, test)
    if not np.array_equal(dixon_345imgobj.header.get_zooms(),dixon_575imgobj.header.get_zooms()):
        CAUTION=True
        if DEBUG: print('CAUTION: dixon_345 and dixon_575 image resolutions are different for '+DIR)

    if RUNTIME_PARAMS['modalities']==modalities_dixon_345_460_575:
        if filename.replace('data/','') == 'ibmcmt_p1/p1-010a/nii/0037-Dixon_TE_575_cf.nii.gz':
            pass
        else:
            assert checkDixonImage(dixon_575img), filename+' may be a phase image'

    # Mask selection (consider not using _af which are poor masks)
    if DEBUG:
        print('selecting mask')
    if 'brcalskd' in DIR:
        filename = os.path.join(DIR, 'roi/Dixon345_'+llshortdict[ll]+'_uk_3.nii.gz')
    elif 'dhmn' in DIR:
        filename = os.path.join(DIR, 'roi/'+ll+'_dixon345_AA_3.nii.gz')
    elif 'arimoclomol' in DIR:
        filename = os.path.join(DIR, 'roi/Dixon_TE_345_'+llshortdict[ll]+'_ssal.nii.gz')
    elif 'alscs' in DIR:
        filename = os.path.join(DIR, 'roi/Dixon345_'+llshortdict[ll]+'_jm_3.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'roi/Dixon345_'+llshortdict[ll]+'_as_3.nii.gz')
    elif 'mdacmt' in DIR:
        filename = os.path.join(DIR, 'roi/'+ll+'_dixon345_cj_3.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'roi/'+ll+'_dixon345_cd_3.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'roi/'+ll+'_dixon345_cmd_3.nii.gz')
    elif 'hypopp' in DIR:
        filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_bk.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'acq/ROI/'+llshortdict[ll]+'_dixon345_er_3.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_er_3.nii.gz')
    elif 'ibmcmt_p1' in DIR:
        filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_bk.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_jm_3.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_gs_3.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_me_3.nii.gz')
        if not os.path.exists(filename):
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_af_3.nii.gz')
    elif 'ibmcmt_p' in DIR:
        if ll == 'calf' and DIR.replace('data/','') == 'ibmcmt_p5/p5-027':
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_at_3.nii.gz')
        elif ll == 'thigh' and DIR.replace('data/','') == 'ibmcmt_p2/p2-008':
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_ta_3.nii.gz')
        elif ll == 'calf' and DIR.replace('data/','') == 'ibmcmt_p5/p5-044':
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_at_3.nii.gz')
        else:
            filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_bk.nii.gz')
            if not os.path.exists(filename):
                filename = os.path.join(DIR, 'roi/Dixon345_'+llshortdict[ll]+'_bk.nii.gz')
            if not os.path.exists(filename):
                filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_me_3.nii.gz')
            if not os.path.exists(filename):
                filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_ta_3.nii.gz')
            if not os.path.exists(filename):
                filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_at_3.nii.gz')
            if not os.path.exists(filename):
                filename = os.path.join(DIR, 'acq/ROI/'+ll+'_dixon345_af_3.nii.gz')
    else:
        filename = None

    # if not os.path.exists(filename):
    #    filename=None

    if filename is None:
        maskimg = np.zeros(dixon_345img.shape, dtype=np.uint8)
    else:
        maskimgobj = nib.load(filename)
        maskimg = np.asanyarray(maskimgobj.dataobj)
        if not np.array_equal(dixon_345imgobj.header.get_zooms(), maskimgobj.header.get_zooms()):
            raise Exception('dixon_345 and mask image resolutions are different for '+DIR)

    if ll == 'calf' and DIR.replace('data/','') == 'ibmcmt_p2/p2-042':
        maskimg[:, :, 3] = 0
    if ll == 'calf' and DIR.replace('data/','') == 'ibmcmt_p3/p3-001':
        maskimg[:, :, 2] = 0
    if ll == 'calf' and DIR.replace('data/','') == 'brcalskd/BRCALSKD_011A':
        maskimg[:, :, 5] = 0
    if ll == 'calf' and DIR.replace('data/','') == 'ibmcmt_p5/p5-061':
        maskimg[:, :, 4] = 0
    if ll == 'calf' and DIR.replace('data/','') == 'ibmcmt_p5/p5-068':
        maskimg[:, :, 5] = 0
    if ll == 'calf' and DIR.replace('data/','') == 'brcalskd/BRCALSKD_002C':
        maskimg[:, :, 6] = 0

    if not multiclass:
        if filename is not None and not convertMRCCentreMaskToBinary(DIR, ll, maskimg):
            raise Exception('convertMRCCentreMaskToBinary returned False')
    else:
        if filename is not None and not convertMRCCentreMaskToStandard(DIR, ll, maskimg):
            raise Exception('convertMRCCentreMaskToStandard returned False')

    return (dixon_460img, dixon_345img, dixon_575img, maskimg, CAUTION)


def load_case(inputdir, DIR, multiclass, test=False):
    try:
        (fatimg, dixon_345img, dixon_575img, maskimg, CAUTION) = load_case_base(inputdir, DIR, multiclass, test)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())
        print('Could not get image data for '+DIR)
        return None

    assert (fatimg.shape == dixon_345img.shape)
    assert (fatimg.shape == dixon_575img.shape)
    assert (fatimg.shape == maskimg.shape)

    t = type(maskimg[0, 0, 0])
    if t is not np.uint8 and t is not np.uint16:
        if DIR.replace('data/','') in ['calf^ibmcmt_p5/p5-034']:
            pass
        else:
            raise Exception('dtype not uint8/16 for mask '+DIR+' it is, '+str(t))

    maskimg = maskimg.astype(np.uint8)

    return DIR, fatimg, dixon_345img, dixon_575img, maskimg, CAUTION


def load_data(DIRS, test=False):
    print('load_data() RUNTIME_PARAMS', RUNTIME_PARAMS)
    start_time = time.time()
    if len(DIRS) < 1:
        raise Exception('No data to load')

    print('Reading %d item(s)...' % len(DIRS))
    n_jobs = max(1, multiprocessing.cpu_count()//2)

    ret = Parallel(n_jobs=n_jobs, verbose=2)(delayed(load_case)(RUNTIME_PARAMS['inputdir'], DIR, RUNTIME_PARAMS['multiclass'], test) for DIR in DIRS)

    X_DIR, X_fatimg, X_dixon_345img, X_dixon_575img, X_maskimg, X_CAUTION = zip(*ret)

    print('Read data time: {} seconds'.format(round(time.time() - start_time, 2)))
    
    if np.sum(X_CAUTION)>0: RUNTIME_PARAMS['caution'] = True

    return list(filter(lambda x: x is not None, X_DIR)), \
        list(filter(lambda x: x is not None, X_fatimg)), \
        list(filter(lambda x: x is not None, X_dixon_345img)), \
        list(filter(lambda x: x is not None, X_dixon_575img)), \
        list(filter(lambda x: x is not None, X_maskimg))


def scale_to_size(img, target_size_y, target_size_x):
    if RUNTIME_PARAMS['multiclass']:
        print(img.dtype,np.unique(img))
        return scale2D(img, target_size_y, target_size_x, order=0, mode='nearest')
    else:
        return scale2D(img, target_size_y, target_size_x, order=3, mode='nearest')


def scale_to_target(img):
    assert (len(img.shape) == 2)

    if np.array_equal(img.shape, [target_size_y, target_size_x]):
        return img

    return scale_to_size(img, target_size_y, target_size_x)


def scale_A_to_B(A, B):
    assert (len(A.shape) == 2)
    assert (len(B.shape) == 2)

    if np.array_equal(A.shape, B.shape):
        return A

    return scale_to_size(A, B.shape[0], B.shape[1])


def read_and_normalize_data(DIRS, test=False):
    DIR, fatimg, dixon_345img, dixon_575img, maskimg = load_data(DIRS, test)
    if len(DIR) < 1:
        raise Exception('No data loaded')

    DIR_new, fatimg_new, dixon_345img_new, dixon_575img_new, maskimg_new = [], [], [], [], []
    for imgi in range(0, len(DIR)):
        for slice in range(0, fatimg[imgi].shape[2]):
            mask_validity = valid_mask(scale_to_target(maskimg[imgi][:, :, slice]), DIR[imgi]+'^slice'+str(slice))
            TO_ADD = False
            if mask_validity == MASK_VALIDITY_VALID:
                TO_ADD = True
            elif mask_validity == MASK_VALIDITY_SINGLESIDED:
                half_size = maskimg[imgi].shape[0]//2
                if valid_mask(scale_to_target(maskimg[imgi][:half_size, :, slice]), DIR[imgi]+'^slice'+str(slice)+'^side1') in [MASK_VALIDITY_VALID, MASK_VALIDITY_SINGLESIDED]:
                    for img_it in (fatimg, dixon_345img, dixon_575img, maskimg):
                        img_it[imgi][:, :, slice] = scale_A_to_B(
                            img_it[imgi][:half_size, :, slice], img_it[imgi][:, :, slice])
                    TO_ADD = True
                elif valid_mask(scale_to_target(maskimg[imgi][half_size:, :, slice]), DIR[imgi]+'^slice'+str(slice)+'^side2') in [MASK_VALIDITY_VALID, MASK_VALIDITY_SINGLESIDED]:
                    for img_it in (fatimg, dixon_345img, dixon_575img, maskimg):
                        img_it[imgi][:, :, slice] = scale_A_to_B(
                            img_it[imgi][half_size:, :, slice], img_it[imgi][:, :, slice])
                    TO_ADD = True
                else:
                    assert(False)
            else:
                TO_ADD = test

            if TO_ADD:
                fatslice = scale_to_target(fatimg[imgi][:, :, slice])
                maskslice = scale_to_target(maskimg[imgi][:, :, slice])
                dixon_345slice = scale_to_target(dixon_345img[imgi][:, :, slice])
                dixon_575slice = scale_to_target(dixon_575img[imgi][:, :, slice])

                if not RUNTIME_PARAMS['multiclass']:
                    if test:
                        assert(np.array_equal(np.unique(maskslice), [0, 1]) or np.array_equal(np.unique(maskslice), [0]))
                    else:
                        assert(np.array_equal(np.unique(maskslice), [0, 1]))
                else:
                    if RUNTIME_PARAMS['al'] == 'calf':
                        valid_values = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17]  
                        valid_values1 = [0, 1, 2, 3, 4, 5, 6, 7]  
                        valid_values2 = [0, 11, 12, 13, 14, 15, 16, 17]  
                    else:
                        valid_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                        valid_values1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                        valid_values2 = [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                        
                    ok=False
                    if mask_validity==MASK_VALIDITY_SINGLESIDED:
                        if (np.array_equal(np.unique(maskslice), valid_values1) or
                                np.array_equal(np.unique(maskslice), valid_values2)):
                            ok = True
                    elif test:
                        if (np.array_equal(np.unique(maskslice), valid_values) or
                                np.array_equal(np.unique(maskslice), [0])
                        ): ok = True
                    else:
                        if np.array_equal(np.unique(maskslice), valid_values): ok = True

                    if not ok:
                        if DEBUG: print(f'DEBUG: np.unique(maskslice)={np.unique(maskslice)} for {DIR}, slice={slice}, test={test}')
                        continue

                fatimg_new.append(fatslice)
                dixon_345img_new.append(dixon_345slice)
                dixon_575img_new.append(dixon_575slice)
                maskimg_new.append(maskslice)
                DIR_text = DIR[imgi]+'^slice'+str(slice)
                DIR_new.append(DIR_text)

    if not test and len(DIR_new) != len(DIR):
        print('INFO: len(DIR_new)!=len(DIR)', len(DIR_new), len(DIR))

    DIR = DIR_new

    fatimg = np.array(fatimg_new, dtype=np.float32)
    dixon_345img = np.array(dixon_345img_new, dtype=np.float32)
    dixon_575img = np.array(dixon_575img_new, dtype=np.float32)
    maskimg = np.array(maskimg_new, dtype=np.uint8)

    del DIR_new, fatimg_new, dixon_345img_new, dixon_575img_new, maskimg_new

    print('fatimg shape and type', fatimg.shape, fatimg.dtype)
    print('dixon_345img shape and type', dixon_345img.shape, dixon_345img.dtype)
    print('dixon_575img shape and type', dixon_575img.shape, dixon_575img.dtype)
    print('maskimg shape and type', maskimg.shape, maskimg.dtype)

    fatimg = fatimg.reshape(fatimg.shape[0], fatimg.shape[1], fatimg.shape[2], 1)
    dixon_345img = dixon_345img.reshape(dixon_345img.shape[0], dixon_345img.shape[1], dixon_345img.shape[2], 1)
    dixon_575img = dixon_575img.reshape(dixon_575img.shape[0], dixon_575img.shape[1], dixon_575img.shape[2], 1)
    maskimg = maskimg.reshape(maskimg.shape[0], maskimg.shape[1], maskimg.shape[2], 1)

    data = np.concatenate((fatimg, dixon_345img, dixon_575img), axis=3)

    print('Data shape:', data.shape)
    print('Mask shape:', maskimg.shape)

    print('Mean data before normalisation: '+str(np.mean(data)))
    print('Std data before normalisation: '+str(np.std(data)))

    if True:
        training_means = np.nanmean(data, axis=(1, 2))
        training_stds = np.nanstd(data, axis=(1, 2))
        training_stds_to_divide_by = training_stds.copy()
        training_stds_to_divide_by[training_stds_to_divide_by == 0] = 1.0

        print('Data means matrix shape: ', training_means.shape)

        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[3]):
                data[i, :, :, j] = (data[i, :, :, j] - training_means[i, j]) / training_stds_to_divide_by[i, j]
    else:
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[3]):
                data[i, :, :, j] -= np.nanmin(data[i, :, :, j])
                data[i, :, :, j] /= np.nanmax(data[i, :, :, j])
                # also try: clip negatives

    print('Mean data after normalisation: '+str(np.mean(data)))
    print('Std data after normalisation: '+str(np.std(data)))

    return DIR, data, maskimg


def MYNET():
    if RUNTIME_PARAMS['multiclass']:
        activation = 'softmax'
        encoder_name='inceptionv4' # inceptionv4 - 41M
        encoder_weights='imagenet+background' # imagenet
        
    else:
        activation = 'sigmoid'
        encoder_name='resnext50_32x4d' # resnext50_32x4d
        encoder_weights='imagenet' # imagenet

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_depth=5-1,
        decoder_channels=(128, 64, 32, 16),
        decoder_use_batchnorm=True,
        in_channels=3,
        classes=RUNTIME_PARAMS['classes'],
        activation=activation
    )


def calc_dice(test_id, test_mask, preds):
    if type(test_id) == str:
        test_id = [test_id]
        test_mask = [test_mask]
        preds = [preds]

    assert (type(test_id) == list)

    DSCs = []
    cutoffs = []
    for case_i in range(0, len(test_id)):
        if MASK_VALIDITY_VALID != valid_mask(np.squeeze(test_mask[case_i]), test_id[case_i]):
            if DEBUG:
                print('DSC: None for %s as it has no valid ground truth' % (test_id[case_i]))
            continue

        cutoffspace = np.linspace(0, 1, 100)
        DSCspace = []
        for i in range(0, len(cutoffspace)):
            binpreds = preds[case_i].copy()
            binpreds[np.where(preds[case_i] > cutoffspace[i])] = 1
            binpreds[np.where(preds[case_i] <= cutoffspace[i])] = 0
            DSCspace.append(numpy_dice_coefficient(test_mask[case_i], binpreds, smooth=1.))

        bestDSC = np.max(DSCspace)
        bestcutoff = cutoffspace[np.argmax(DSCspace)]
        if DEBUG:
            print('DSC: %f at cut-off %f for %s' % (bestDSC, bestcutoff, test_id[case_i]))
        DSCs.append(bestDSC)
        cutoffs.append(bestcutoff)

        calc_dice_file = 'calc_dice_%s_%s.csv' % (RUNTIME_PARAMS['inputdir'].replace('/', '_'), RUNTIME_PARAMS['al'])
        if 'calc_dice_num_calls' not in RUNTIME_PARAMS:
            RUNTIME_PARAMS['calc_dice_num_calls'] = 0
            with open(calc_dice_file, 'wt') as outfile:
                outfile.write('id,DSC,best_cutoff\n')

        RUNTIME_PARAMS['calc_dice_num_calls'] += 1
        with open(calc_dice_file, 'at') as outfile:
            outfile.write('%s,%f,%f\n' % (test_id[case_i], bestDSC, bestcutoff))

    if len(DSCs) > 0:
        meanDSC = np.mean(DSCs)
    else:
        meanDSC = None

    if len(test_id) > 1 and DEBUG:
        print('meanDSC: %s' % (str(meanDSC)))

    return DSCs, cutoffs


def print_scores(data, data_mask, preds, std_preds, test_id):
    print(data.shape, data.dtype)
    print(data_mask.shape, data_mask.dtype)
    print(preds.shape, preds.dtype)
    print(len(test_id))

    print('Saving results')
    for i in range(0, preds.shape[0]):
        DIR = test_id[i]
        # DSCs,cutoffs=calc_dice(DIR,np.squeeze(data_mask[i]),preds[i])
        # assert(len(DSCs) in [0,1])

        # if len(DSCs)==1:
        #    results_DSC_file='results_DSC_%s_%s.csv'%(RUNTIME_PARAMS['inputdir'].replace('/','_'),RUNTIME_PARAMS['al'])
        #    if 'print_scores_num_calls' not in RUNTIME_PARAMS:
        #        RUNTIME_PARAMS['print_scores_num_calls']=0
        #        with open(results_DSC_file,'wt') as outfile:
        #            outfile.write('id,DSC,best_cutoff\n')

        #    RUNTIME_PARAMS['print_scores_num_calls']+=1
        #    with open(results_DSC_file,'at') as outfile:
        #        outfile.write('%s,%f,%f\n'%(DIR,DSCs[0],cutoffs[0]))

        TK = DIR.split('^')
        assert (len(TK) == 3)
        ll = TK[0]
        DIR = TK[1]
        slice = int(TK[2].replace('slice', ''))

        tag = 'parcellation' if RUNTIME_PARAMS['multiclass'] else 'segmentation'
        filename = "%s/%s_%s.nii.gz" % (DIR, ll, tag)
        filename_std = "%s/%s_%s_var.nii.gz" % (DIR, ll, tag)

        if RUNTIME_PARAMS['caution']:
            filename=filename.replace('.nii.gz','-caution.nii.gz')
            filename_std=filename_std.replace('.nii.gz','-caution.nii.gz')
            
        maskimg = None
        maskimg_std = None

        if slice > 0:
            try:
                nibobj = nib.load(filename)
                maskimg = nibobj.get_fdata().astype(np.float32)
            except:
                print('Could not load '+filename)
            try:
                nibobj_std = nib.load(filename_std)
                maskimg_std = nibobj_std.get_fdata().astype(np.float32)
            except:
                print('Could not load '+filename_std)

        if maskimg is None or maskimg_std is None:
            if 'Amy_GOSH' in DIR:
                fatfilename = glob.glob(os.path.join(DIR, '*DIXON_F.nii*'))[0]
            elif RUNTIME_PARAMS['inputdir'] != 'train' and RUNTIME_PARAMS['inputdir'] != 'validate':
                if RUNTIME_PARAMS['modalities']==modalities_t1:
                    fatfilename = glob.glob(os.path.join(DIR, 't1.nii*'))[0]
                elif RUNTIME_PARAMS['modalities']==modalities_t2_stir:
                    fatfilename = glob.glob(os.path.join(DIR, 't2_stir.nii*'))[0]
                else:
                    fatfilename = glob.glob(os.path.join(DIR, 'fat.nii*'))[0]
            else:
                fatfilename = os.path.join(DIR, 'ana/fatfraction/'+ll+'/fat.nii.gz')
                if not os.path.exists(fatfilename):
                    fatfilename = os.path.join(DIR, 'ana/fatfraction/'+llshortdict[ll]+'/fat.nii.gz')

            nibobj = nib.load(fatfilename)
            nibobj_std = nib.load(fatfilename)
            shape = nibobj.get_fdata().shape[0:3]
            maskimg = np.zeros(shape, dtype=np.float32)
            maskimg_std = np.zeros(shape, dtype=np.float32)

        assert (slice >= 0)
        assert (slice < maskimg.shape[2])

        if RUNTIME_PARAMS['multiclass']:
            axis = 0
            assert (preds[i].shape[axis] == RUNTIME_PARAMS['classes'])
            img_to_save = scale_to_size(np.argmax(preds[i], axis=axis), maskimg.shape[0], maskimg.shape[1])
            std_img_to_save = scale_to_size(np.argmax(std_preds[i], axis=axis), maskimg.shape[0], maskimg.shape[1])
        else:
            img_to_save = scale_to_size(preds[i,0], maskimg.shape[0], maskimg.shape[1])
            std_img_to_save = scale_to_size(std_preds[i,0], maskimg.shape[0], maskimg.shape[1])

        maskimg[:, :, slice] = img_to_save
        maskimg_std[:, :, slice] = std_img_to_save

        if DEBUG:
            print('Saving '+filename+' (slice %d)' % (slice))
        save_nifti(maskimg, nibobj.affine, nibobj.header, filename)

        if DEBUG:
            print('Saving '+filename_std+' (slice %d)' % (slice))
        save_nifti(maskimg_std, nibobj_std.affine, nibobj_std.header, filename_std)


def saveTrainingMetrics(history, label, filename):
    plt_x = list(range(1, len(history['loss'])+1))
    fig = plt.figure(figsize=(12, 5), dpi=100)
    plt.subplot(121)
    plt.plot(plt_x, history['loss'], label='loss')
    plt.plot(plt_x, history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(label)
    plt.subplot(122)
    plt.plot(plt_x, history['acc'], label='acc')
    plt.plot(plt_x, history['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.legend()

    ep = np.argmin(history['val_loss'])
    infostr = 'val_loss %.4f@%d, val_acc %.4f' % (history['val_loss'][ep],
                                                       ep+1, history['val_acc'][ep])

    plt.title(infostr)
    plt.savefig(filename)
    plt.close(fig)


def augmentData(batch_images, mask_images):
    for batchi in range(batch_images.shape[0]):
        TAG_augment_crop_and_resize = True
        if TAG_augment_crop_and_resize and np.random.randint(0, 2) == 0:
            if DEBUG:
                fig = plt.figure()
                plt.subplot(221)
                plt.imshow(batch_images[batchi, 0, :, :], cmap='gray')
                plt.subplot(222)
                plt.imshow(mask_images[batchi, 0, :, :], cmap='gray')
            croparea = [0, 0, batch_images.shape[2]//2, batch_images.shape[3]]
            if np.random.randint(0, 2) == 1:
                croparea[0] = batch_images.shape[2]//2
            batch_images[batchi] = resized_crop(batch_images[batchi:batchi+1], *croparea,
                                                batch_images.shape[2:4], tt.InterpolationMode.BILINEAR
                                                )
            mask_images[batchi] = resized_crop(mask_images[batchi:batchi+1], *croparea,
                                               batch_images.shape[2:4], tt.InterpolationMode.NEAREST
                                               )
            if DEBUG:
                plt.subplot(223)
                plt.imshow(batch_images[batchi, 0, :, :], cmap='gray')
                plt.subplot(224)
                plt.imshow(mask_images[batchi, 0, :, :], cmap='gray')
                plt.savefig('__sample_augment_crop_and_resize.png')
                plt.close(fig)

        # did not work/help: rot90(k=2), flip_left_right, flip_up_down (would invalidate L/R orientation)

    return batch_images, mask_images


class MMSegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return {
            'image': np.transpose(self.images[index], (2, 0, 1)),
            'mask': np.transpose(self.masks[index], (2, 0, 1))
        }


def train(train_DIRS, test_DIRS, BREAK_OUT_AFTER_FIRST_FOLD):
    "Train and validate"
    test_DIR, test_data, test_maskimg = read_and_normalize_data(test_DIRS, True)
    train_DIR, train_data, train_maskimg = read_and_normalize_data(train_DIRS)

    outer_train_subjects = []
    for i in range(0, len(train_DIR)):
        outer_train_subjects.append(get_subject_id_from_DIR(train_DIR[i]))

    outer_train_subjects = np.array(outer_train_subjects)

    kf = GroupKFold(n_splits=5)
    fold = 0
    for train_index, valid_index in kf.split(train_data, groups=outer_train_subjects):
        if RUNTIME_PARAMS['inputdir'] == 'train':
            print('fold', fold+1)
        else:
            print('outer fold', RUNTIME_PARAMS['outerfold']+1, 'inner fold', fold+1)

        X_train_this, y_train_this = train_data[train_index], train_maskimg[train_index]
        X_valid_this, y_valid_this = train_data[valid_index], train_maskimg[valid_index]
        DIR_valid = list(np.array(train_DIR)[valid_index])

        print('X_train_this', X_train_this.shape, X_train_this.dtype)
        print('X_valid_this', X_valid_this.shape, X_valid_this.dtype)
        print('y_train_this', y_train_this.shape, y_train_this.dtype)
        print('y_valid_this', y_valid_this.shape, y_valid_this.dtype)

        train_subjects = outer_train_subjects[train_index]
        valid_subjects = outer_train_subjects[valid_index]

        common_subjects = np.intersect1d(train_subjects, valid_subjects, assume_unique=False, return_indices=False)
        assert (common_subjects.size == 0)

        print('batch_size: %d' % (RUNTIME_PARAMS['batch_size']))

        if RUNTIME_PARAMS['multiclass']:
            y_train_this = torch.nn.functional.one_hot(torch.LongTensor(
                np.squeeze(y_train_this, axis=3)), RUNTIME_PARAMS['classes'])
            y_valid_this = torch.nn.functional.one_hot(torch.LongTensor(
                np.squeeze(y_valid_this, axis=3)), RUNTIME_PARAMS['classes'])

        train_dataloader = DataLoader(
            MMSegDataset(X_train_this, y_train_this),
            batch_size=RUNTIME_PARAMS['batch_size'],
            shuffle=True,
            num_workers=0
        )

        valid_dataloader = DataLoader(
            MMSegDataset(X_valid_this, y_valid_this),
            batch_size=RUNTIME_PARAMS['batch_size'],
            shuffle=False,
            num_workers=0
        )

        device = RUNTIME_PARAMS['device']
        model = MYNET().to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=RUNTIME_PARAMS['lr'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.8)

        def MMSegLoss(y_pred, y_true):
            loss1 = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=False)
            # loss2 = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.0)
            return loss1.forward(y_pred, y_true)  # + loss2.forward(y_pred, y_true)

        loss_fn = MMSegLoss
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
        for epoch in range(5 if RUNTIME_PARAMS['smoketest'] else 5555):
            epoch_st = time.time()

            model.train()
            losses_this_epoch = []
            accs_this_epoch = []
            with torch.set_grad_enabled(True):
                for data in tqdm(train_dataloader, leave=False, desc='Training'):
                    image, mask = augmentData(data['image'], data['mask'])
                    image = image.to(device)
                    mask = mask.to(device)
                    optimiser.zero_grad()
                    pred = model(image)
                    loss = loss_fn(pred, mask)
                    loss.backward()
                    optimiser.step()
                    losses_this_epoch.append(loss.item())
                    if RUNTIME_PARAMS['multiclass']:
                        pred = torch.argmax(pred,dim=1)
                        mask = torch.argmax(mask,dim=1)
                    else:
                        pred[pred<0.5]=0
                        pred[pred>=0.5]=1
                    accs_this_epoch.append(torch.sum(pred==mask).cpu()/pred.numel())

            history['loss'].append(np.mean(losses_this_epoch))
            history['acc'].append(np.mean(accs_this_epoch))

            model.eval()
            losses_this_epoch = []
            accs_this_epoch = []
            with torch.set_grad_enabled(False):
                for data in tqdm(valid_dataloader, leave=False, desc='Validating'):
                    image, mask = data['image'], data['mask']
                    image = image.to(device)
                    mask = mask.to(device)
                    pred = model(image)
                    loss = loss_fn(pred, mask)
                    losses_this_epoch.append(loss.item())
                    if RUNTIME_PARAMS['multiclass']:
                        pred = torch.argmax(pred,dim=1)
                        mask = torch.argmax(mask,dim=1)
                    else:
                        pred[pred<0.5]=0
                        pred[pred>=0.5]=1
                    accs_this_epoch.append(torch.sum(pred==mask).cpu()/pred.numel())

            history['val_loss'].append(np.mean(losses_this_epoch))
            history['val_acc'].append(np.mean(accs_this_epoch))

            # lr_scheduler.step()

            best_epoch = np.argmin(history['val_loss'])
            epoch_et = time.time() - epoch_st

            metrics_str = ''
            for metric in history:
                metrics_str += f'{metric}: {history[metric][-1]:.4f} '
            if best_epoch == epoch:
                metrics_str += '[*]'
                best_epoch_state_dict = model.state_dict()
            print(f'Epoch {epoch+1} completed in {epoch_et:.1f}s, {metrics_str}')

            if best_epoch < epoch-RUNTIME_PARAMS['patience']:
                print(f"Validation loss has not improved within {RUNTIME_PARAMS['patience']} epoch(s), early stopping")
                break

        model.load_state_dict(best_epoch_state_dict)
        if RUNTIME_PARAMS['inputdir'] == 'train':
            modelfilename = 'models/full.%d.%s.%s.%s.model' % (
                fold, RUNTIME_PARAMS['al'], 'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary', 
                RUNTIME_PARAMS['modalities']
                )
            torch.save(best_epoch_state_dict, modelfilename)
            trainingMetricsFilename = 'full.%d.%s.%s.%s.%s.pdf' % (
                fold, RUNTIME_PARAMS['al'], 'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary', 
                RUNTIME_PARAMS['modalities'], RUNTIME_PARAMS['inputdir']
                )
        else:
            trainingMetricsFilename = 'full.%d.%d.%s.%s.%s.%s.pdf' % (
                RUNTIME_PARAMS['outerfold'], 
                fold, RUNTIME_PARAMS['al'], 'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary', 
                RUNTIME_PARAMS['modalities'], RUNTIME_PARAMS['inputdir']
                )

        saveTrainingMetrics(history, trainingMetricsFilename, os.path.join('metrics',trainingMetricsFilename))

        best_epoch_val_acc = history['val_acc'][best_epoch]
        best_epoch_val_loss = history['val_loss'][best_epoch]
        best_epoch_train_acc = history['acc'][best_epoch]
        best_epoch_train_loss = history['loss'][best_epoch]

        if fold == 0:
            val_accs = [best_epoch_val_acc]
            val_losses = [best_epoch_val_loss]
            train_accs = [best_epoch_train_acc]
            train_losses = [best_epoch_train_loss]
            best_epochs = [best_epoch]
        else:
            val_accs.extend([best_epoch_val_acc])
            val_losses.extend([best_epoch_val_loss])
            train_accs.extend([best_epoch_train_acc])
            train_losses.extend([best_epoch_train_loss])
            best_epochs.append(best_epoch)

        print('Fold %d [%s, %s, batch_size=%d, multiclass=%s, modalities=%s]' % (
                fold+1, 
                RUNTIME_PARAMS['al'], 
                RUNTIME_PARAMS['inputdir'], 
                RUNTIME_PARAMS['batch_size'], 
                RUNTIME_PARAMS['multiclass'],
                RUNTIME_PARAMS['modalities'],
                )
            )
        print('mean best epoch %d +- %d [range %d - %d]' % (np.mean(best_epochs),
              np.std(best_epochs), np.min(best_epochs), np.max(best_epochs)))

        print('mean acc [tra] %.4f +- %.4f [range %.4f - %.4f]' %
              (np.mean(train_accs), np.std(train_accs), np.min(train_accs), np.max(train_accs)))
        print('mean acc [val] %.4f +- %.4f [range %.4f - %.4f]' %
              (np.mean(val_accs), np.std(val_accs), np.min(val_accs), np.max(val_accs)))

        print('mean loss [tra] %.4f +- %.4f [range %.4f - %.4f]' %
              (np.mean(train_losses), np.std(train_losses), np.min(train_losses), np.max(train_losses)))
        print('mean loss [val] %.4f +- %.4f [range %.4f - %.4f]' %
              (np.mean(val_losses), np.std(val_losses), np.min(val_losses), np.max(val_losses)))

        # test
        # p = model.predict(test_data, batch_size=RUNTIME_PARAMS['batch_size'], verbose=1)
        # p = np.expand_dims(p, axis=0)

        # if fold == 0:
        #    preds = p
        # else:
        #    preds = np.concatenate((preds, p), axis=0)
        fold += 1
        if BREAK_OUT_AFTER_FIRST_FOLD:
            break

#    mean_preds=np.nanmean(preds,axis=0)
#    std_preds=np.nanstd(preds,axis=0)
#    DSCs,_cutoffs=calc_dice(test_DIR,test_maskimg,mean_preds)
#    print_scores(test_data,test_maskimg,mean_preds,std_preds,test_DIR)

    return None  # DSCs


def test(test_DIRS):
    print('test', test_DIRS)
    print('RUNTIME_PARAMS', RUNTIME_PARAMS)
    test_DIR, test_data, test_maskimg = read_and_normalize_data(test_DIRS, True)
    
    device = RUNTIME_PARAMS['device']
    model = MYNET().to(device)
    for fold in range(5):
        weightsfile = 'models/full.%d.%s.%s.%s.model' % (
            fold, RUNTIME_PARAMS['al'], 'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary',
            RUNTIME_PARAMS['modalities']
            )
        weightsfile = os.path.join(INSTALL_DIR, weightsfile)

        if not os.path.exists(weightsfile):
            continue
            msg = 'Downloading '+os.path.basename(weightsfile)
            print(msg)
            if RUNTIME_PARAMS['widget'] is not None:
                RUNTIME_PARAMS['widget']['text'] = msg
                RUNTIME_PARAMS['widget'].update()
            url = "https://github.com/bariskanber/musclesenseworkbench/releases/download/r1/%s" % (
                os.path.basename(weightsfile))
            urllib.request.urlretrieve(url, weightsfile)

        if RUNTIME_PARAMS['widget'] is not None:
            RUNTIME_PARAMS['widget']['text'] = 'Calculating mask (%.0f%%)...' % (100*float(fold+1)/5)
            RUNTIME_PARAMS['widget'].update()

        model.load_state_dict(torch.load(weightsfile,map_location=device))

        test_dataloader = DataLoader(
            MMSegDataset(test_data, test_maskimg),
            batch_size=RUNTIME_PARAMS['batch_size'],
            shuffle=False,
            num_workers=0
        )

        p = None
        for data in tqdm(test_dataloader, leave=False, desc='Inferring'):
            image, mask = data['image'], data['mask']
            image = image.to(device)
            mask = mask.to(device)
            p_this = model(image).detach().cpu()
            if p is None:
                p = p_this
            else:
                p = np.concatenate((p, p_this), axis=0)

        p = np.expand_dims(p, axis=0)

        if fold == 0:
            preds = p
        else:
            preds = np.concatenate((preds, p), axis=0)

    mean_preds = np.nanmean(preds, axis=0)
    std_preds = np.nanstd(preds, axis=0)

    # mean_preds-=std_preds*1.96

#    DSCs,_cutoffs=calc_dice(test_DIR,test_maskimg,mean_preds)
    print_scores(test_data, test_maskimg, mean_preds, std_preds, test_DIR)
    return None


def main(al, inputdir, modalities, multiclass, widget):
    RUNTIME_PARAMS['al'] = al
    RUNTIME_PARAMS['inputdir'] = inputdir
    RUNTIME_PARAMS['modalities'] = modalities
    RUNTIME_PARAMS['widget'] = widget
    RUNTIME_PARAMS['multiclass'] = multiclass  # individual muscle segmentation (vs. whole muscle)

    if RUNTIME_PARAMS['widget'] is not None:
        RUNTIME_PARAMS['widget']['text'] = 'Calculating mask...'
        RUNTIME_PARAMS['widget'].update()

    RUNTIME_PARAMS['device'] = None
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        gb = 1024.0*1024*1024
        #print(f'total    : {info.total/gb}')
        #print(f'free     : {info.free/gb}')
        #print(f'used     : {info.used/gb}')   
        if info.free/gb<4:
            print('CUDA is low on memory')
        else:
            RUNTIME_PARAMS['device'] = torch.device('cuda:0')

    if RUNTIME_PARAMS['device'] is None:
        RUNTIME_PARAMS['device'] = torch.device('cpu')
        print('CUDA not available, will run on CPU')

    if RUNTIME_PARAMS['smoketest']:
        print('Smoke test enabled')
        time.sleep(5)

    if RUNTIME_PARAMS['multiclass']:
        RUNTIME_PARAMS['classes'] = 17+1 if RUNTIME_PARAMS['al'] == 'calf' else 31+1
    else:
        RUNTIME_PARAMS['classes'] = 1

    ll = RUNTIME_PARAMS['al']

    start_time = time.time()

    if RUNTIME_PARAMS['inputdir'] == 'train' or RUNTIME_PARAMS['inputdir'] == 'validate':
        torch.manual_seed(44)
        random.seed(44)
        np.random.seed(44)
        torch.cuda.manual_seed(44)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        DIRS = []
        for DATA_DIR in ['arimoclomol','mdacmt','alscs','dhmn','brcalskd', 'hypopp', 'ibmcmt_p1', 'ibmcmt_p2', 'ibmcmt_p3', 'ibmcmt_p4', 'ibmcmt_p5', 'ibmcmt_p6']:
            for de in glob.glob(os.path.join('data/'+DATA_DIR, '*')):
                if not os.path.isdir(de):
                    if DEBUG:
                        print(de+' is not a directory')
                    continue

                #if not os.path.isdir(os.path.join(de, 'ana/fatfraction/'+ll)) and not os.path.isdir(os.path.join(de, 'ana/fatfraction/'+llshortdict[ll])):
                #    if DEBUG:
                #        print(de+' does not have a '+ll+' directory')
                #    continue
                
                if RUNTIME_PARAMS['modalities']==modalities_t1:
                    files = glob.glob(os.path.join(de, 'nii/t1w_%s_*.nii*'%(ll)))
                    if len(files)==0:
                        files = glob.glob(os.path.join(de, 'nii/*-T1_*_%s.nii.gz'%(llshortdict[ll].upper())))
                    if len(files)==0: 
                        if DEBUG:
                            print(de+' does not have a T1 image')
                        continue                
                elif RUNTIME_PARAMS['modalities']==modalities_t2_stir:
                    files = glob.glob(os.path.join(de, 'nii/stir_%s_*.nii*'%(ll)))
                    if len(files)==0:
                        files = glob.glob(os.path.join(de, 'nii/*_STIR_%s.nii.gz'%(llshortdict[ll].upper())))
                    if len(files)==0:                
                        if DEBUG:
                            print(de+' does not have a STIR image')
                        continue                

                files = glob.glob(os.path.join(de, 'roi/?ixon*345_'+ll+'_*.nii.gz'))
                if len(files) == 0:
                    files = glob.glob(os.path.join(de, 'roi/?ixon*345_'+llshortdict[ll]+'_*.nii.gz'))
                if len(files) == 0:
                    files = glob.glob(os.path.join(de, 'roi/'+ll+'_?ixon*345_*.nii.gz'))
                if len(files) == 0:
                    files = glob.glob(os.path.join(de, 'roi/'+llshortdict[ll]+'_?ixon*345_*.nii.gz'))
                if len(files) == 0:
                    files = glob.glob(os.path.join(de, 'acq/ROI/'+ll+'_*.nii.gz'))
                if len(files) == 0:
                    files = glob.glob(os.path.join(de, 'acq/ROI/'+llshortdict[ll]+'_*.nii.gz'))
                if len(files) == 0:
                    if DEBUG:
                        print(de+' does not have a '+ll+' mask')
                    continue

                files = glob.glob(os.path.join(de, 'nii/*_lg_*.nii.gz'))
                for f in files:
                    if DEBUG:
                        print('Renaming %s as %s'%(f, f.replace('_lg_', '_')))
                    os.rename(f, f.replace('_lg_', '_'))

                if False and ll == 'calf' and de.replace('data/','') in [
                        'hypopp/015_b', 'hypopp/016_b', 'hypopp/017_b', 'hypopp/019_b', 'hypopp/022_b', 'ibmcmt_p2/p2-046', 'ibmcmt_p2/p2-026', 'ibmcmt_p4/p4-008', 'ibmcmt_p4/p4-029', 'ibmcmt_p4/p4-030', 'ibmcmt_p3/p3-030', 'ibmcmt_p4/p4-054', 'ibmcmt_p4/p4-071', 'ibmcmt_p5/p5-033', 'ibmcmt_p3/p3-046', 'ibmcmt_p5/p5-042', 'ibmcmt_p5/p5-044', 'ibmcmt_p5/p5-049', 'ibmcmt_p5/p5-060', 'ibmcmt_p6/p6-008', 'ibmcmt_p5/p5-032', 'ibmcmt_p6/p6-032', 'ibmcmt_p5/p5-039', 'ibmcmt_p6/p6-044', 'ibmcmt_p6/p6-062', 'ibmcmt_p4/p4-046', 'ibmcmt_p4/p4-049']:
                    if DEBUG:
                        print('skipping %s that has L/R segmented on different slices or only one side segmented' % (de))
                    continue

                if ll == 'thigh' and de.replace('data/','') in [
                    'ibmcmt_p3/p3-072',  # mask is out-of-alignment
                    'ibmcmt_p5/p5-004',  # looks a bit off
                ]:
                    if DEBUG:
                        print('skipping %s that has bad mask' % (de))
                    #assert(False)
                    continue

                if False and ll == 'calf' and de.replace('data/','') in [
                    'ibmcmt_p1/p1-001b',  # looks quite badly segmented
                ]:
                    if DEBUG:
                        print('skipping %s that has bad mask' % (de))
                    continue

                if False and ll == 'calf' and de.replace('data/','') in [
                    'ibmcmt_p2/p2-065',  # actually might be good to include
                    'ibmcmt_p3/p3-047',  # actually looks good
                    'ibmcmt_p3/p3-055',  # actually looks good
                    'ibmcmt_p3/p3-008',  # very noisy but add (at least to train)
                    'ibmcmt_p5/p5-034',  # very bad but add (at least to train)
                ]:
                    if DEBUG:
                        print('skipping %s that have poor image quality' % (de))
                    continue

                DIRS.append(ll+'^'+de)

                # if False and len(DIRS)>20: break

        if RUNTIME_PARAMS['smoketest']:
            DIRS = DIRS[:20]
        print('%d cases found' % (len(DIRS)))
        #print(DIRS)

        if RUNTIME_PARAMS['inputdir'] == 'train':
            train_DIRS = sorted(DIRS[:])
            if RUNTIME_PARAMS['al'] == 'calf':
                test_DIRS = [
                    'calf^data/ibmcmt_p2/p2-008',
                ]
            else:
                test_DIRS = [
                    'thigh^data/ibmcmt_p1/p1-007b',
                ]

            train_DIRS = np.array(train_DIRS)
            test_DIRS = np.array(test_DIRS)

            np.random.shuffle(train_DIRS)

            train(train_DIRS, test_DIRS, BREAK_OUT_AFTER_FIRST_FOLD=False)
        elif RUNTIME_PARAMS['inputdir'] == 'validate':
            print('Validate mode')
            print('Use this mode to perform nested cross validation over all the available training data')
            print('')
            outloop_req = input('Input outer loop number (default=all): ').strip()
            if len(outloop_req) == 0:
                outloop_req = 'all'
            print(outloop_req)

            difficult_cases = [
                #        'thigh^data/ibmcmt_p1/p1-007a',
                #        'thigh^data/ibmcmt_p1/p1-007b',
                #        'thigh^data/ibmcmt_p2/p2-008',
                #        'thigh^data/ibmcmt_p2/p2-008b',
                #        'calf^data/ibmcmt_p4/p4-044',
                #        'calf^data/ibmcmt_p4/p4-061',
                #        'calf^data/hypopp/006_b',
                #        'calf^data/ibmcmt_p2/p2-008',
                #        'calf^data/ibmcmt_p2/p2-030',
                #        'calf^data/ibmcmt_p2/p2-030b',
                #        'calf^data/ibmcmt_p2/p2-008b',
            ]

            DSCs = []

            case_i = 0
            for difficult_case in difficult_cases:
                case_i += 1
                print('Doing difficult case %d of %d with LOO' % (case_i, len(difficult_cases)))

                train_DIRS = DIRS[:]
                test_DIRS = [difficult_case]

                if difficult_case in train_DIRS:
                    train_DIRS.remove(difficult_case)

                train_DIRS = np.array(train_DIRS)
                test_DIRS = np.array(test_DIRS)

                np.random.shuffle(train_DIRS)

                DSCarray = train(train_DIRS, test_DIRS, BREAK_OUT_AFTER_FIRST_FOLD=False)
                DSCs.extend(DSCarray)

                print('batch_size', RUNTIME_PARAMS['batch_size'])
                print('difficult case %d or %d' % (case_i, len(difficult_cases)))
                if len(DSCs) > 0:
                    print('mean DSC %.4f +- %.4f [n=%d]' % (np.mean(DSCs), np.std(DSCs), len(DSCs)))
                    print('range DSC %.4f - %.4f' % (np.min(DSCs), np.max(DSCs)))

            if True:
                DIRS = np.array(DIRS)

                np.random.shuffle(DIRS)

                subjects = []
                for i in range(0, len(DIRS)):
                    subjects.append(get_subject_id_from_DIR(DIRS[i]))

                subjects = np.array(subjects)

                kf = GroupKFold(n_splits=5)
                fold = 0

                for train_index, test_index in kf.split(DIRS, groups=subjects):
                    print('outer fold', fold+1)
                    RUNTIME_PARAMS['outerfold'] = fold

                    if outloop_req != 'all' and fold+1 != int(outloop_req):
                        fold += 1
                        continue

                    train_DIRS = DIRS[train_index]
                    test_DIRS = DIRS[test_index]

                    train_subjects = subjects[train_index]
                    test_subjects = subjects[test_index]

                    common_subjects = np.intersect1d(train_subjects, test_subjects,
                                                     assume_unique=False, return_indices=False)
                    assert (common_subjects.size == 0)

                    print('Removing already tested cases from test set')
                    train_DIRS = list(train_DIRS)
                    test_DIRS = list(test_DIRS)

                    for DIR in difficult_cases:
                        if DIR in test_DIRS:
                            test_DIRS.remove(DIR)
                            train_DIRS.append(DIR)

                    train_DIRS = np.array(train_DIRS)
                    test_DIRS = np.array(test_DIRS)

                    np.random.shuffle(train_DIRS)

                    DSCarray = train(train_DIRS, test_DIRS,
                                     BREAK_OUT_AFTER_FIRST_FOLD=True)
                    DSCs.extend(DSCarray)

                    print('batch_size', RUNTIME_PARAMS['batch_size'])
                    print('outer fold', fold+1)
                    print('mean DSC %.4f +- %.4f [n=%d]' % (np.mean(DSCs), np.std(DSCs), len(DSCs)))
                    print('range DSC %.4f - %.4f' % (np.min(DSCs), np.max(DSCs)))

                    fold += 1
    else:
        DIRS = []
        for DATA_DIR in [RUNTIME_PARAMS['inputdir']]:
            for de in glob.glob(os.path.join(DATA_DIR, '*')):
                if not os.path.isdir(de):
                    if DEBUG:
                        print(de+' is not a directory')
                    continue

                DIRS.append(ll+'^'+de)

        if len(DIRS) == 0:
            DIRS.append(ll+'^'+RUNTIME_PARAMS['inputdir'])

        print(DIRS)
        print('%d cases found' % (len(DIRS)))

        DIRS = np.array(DIRS)
        DSCarray = test(DIRS)

    print('Running time: {} hours'.format(round((time.time() - start_time)/3600.0, 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("mmseg_ll")
    parser.add_argument('-al', type=str, help='anatomical location (calf/thigh)')
    parser.add_argument('-inputdir', type=str, help='input directory/folder (or train/validate)')
    parser.add_argument('-modalities', type=str, help='input modalities (one of %s; default: %s)'%(', '.join(available_modalities),
                            available_modalities[0]), default=available_modalities[0])
    parser.add_argument('--multiclass', action="store_true", help='individual muscle segmentation (vs. whole muscle)')
    parser.add_argument('--version', action='version', version='1.1.0')
    args = parser.parse_args()

    if args.inputdir is None or args.al is None:
        parser.print_help()
        sys.exit(1)

    main(args.al, args.inputdir, args.modalities, args.multiclass, widget=None)
