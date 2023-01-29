#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import nibabel as nib
import sys
import glob
import time
import math
import string
import random
import multiprocessing
import json
from joblib import Parallel, delayed
import shutil
import argparse
import urllib.request
import skimage.measure
from sklearn.model_selection import GroupKFold

import tensorflow as tf
import tensorflow.keras
tensorflow.keras.backend.set_image_data_format('channels_last')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from convertMRCCentreMaskToBinary import convertMRCCentreMaskToBinary
from convertMRCCentreMaskToStandard import convertMRCCentreMaskToStandard
from utils import numpy_dice_coefficient, scale2D, saveTrainingMetrics
from nifti_tools import save_nifti
from noisy import noisy

import segmentation_models as sm
sm.set_framework('tf.keras')

import socket
MY_PC=1 if socket.gethostname()=="bkanber-gpu" else 0

if MY_PC:
    import matplotlib.pyplot as plt

DEBUG=True
llshortdict={'thigh':'th','calf':'cf'}

scale_down_factor=1
simplerModel=False

np.random.seed(0)

target_size_y,target_size_x=320//scale_down_factor,160//scale_down_factor

if MY_PC: batch_size=1
else: 
    batch_size=1

RUNTIME_PARAMS=dict()

MODULE_NAME=os.path.basename(__file__)
INSTALL_DIR=os.path.dirname(os.path.realpath(__file__))
print('INSTALL_DIR',INSTALL_DIR)
print('MODULE_NAME',MODULE_NAME)

if False:
    with open('%s/.keras/keras.json'%(os.path.expanduser("~")),'rt') as json_file:
        data = json.load(json_file)
        if DEBUG: print(data)

        assert(data['floatx']=="float32")
        assert(data['epsilon']==1e-07)
        assert(data['backend']=="tensorflow")
        assert(data['image_data_format']=="channels_last")
        assert(data['image_dim_ordering']=="th")

def get_subject_id_from_DIR(DIR):
        DIR=DIR.split('^') # e.g. thigh^brcalskd/BRCALSKD_056C
        assert(len(DIR)>=2)
        DIR=DIR[1]
        
        if 'BRCALSKD' in DIR: 
            subject=os.path.basename(DIR).replace('BRCALSKD_','')
            assert(len(subject)==4)
            return 'BRCALSKD.'+subject[:3]   

        if 'HYPOPP' in DIR.upper():
            subject=os.path.basename(DIR)
            assert(len(subject)==5)
            assert(subject[3]=='_')
            return 'HYPOPP.'+subject[:3]

        if 'ibmcmt_p1' in DIR:
            subject=os.path.basename(DIR).replace('p1-','')
            assert(len(subject)==4)
            return 'ibmcmt_p1.'+subject[:3]   

        if 'ibmcmt_p2' in DIR:
            subject=os.path.basename(DIR).replace('p2-','')
            assert(len(subject)==4 or len(subject)==3)
            return 'ibmcmt_p2-6.'+subject[:3]   

        if 'ibmcmt_p3' in DIR:
            subject=os.path.basename(DIR).replace('p3-','')
            assert(len(subject)==4 or len(subject)==3)
            return 'ibmcmt_p2-6.'+subject[:3]   

        if 'ibmcmt_p4' in DIR:
            subject=os.path.basename(DIR).replace('p4-','')
            assert(len(subject)==4 or len(subject)==3)
            return 'ibmcmt_p2-6.'+subject[:3]   

        if 'ibmcmt_p5' in DIR:
            subject=os.path.basename(DIR).replace('p5-','')
            assert(len(subject)==4 or len(subject)==3)
            return 'ibmcmt_p2-6.'+subject[:3]   

        if 'ibmcmt_p6' in DIR:
            subject=os.path.basename(DIR).replace('p6-','')
            assert(len(subject)==4 or len(subject)==3)
            return 'ibmcmt_p2-6.'+subject[:3]   

        raise Exception('Unable to determine subject ID')

MASK_VALIDITY_VALID=1
MASK_VALIDITY_BLANKMASK=2
MASK_VALIDITY_BAD=3
MASK_VALIDITY_SINGLESIDED=4

def valid_mask(mask_original,help_str):
    mask=mask_original.copy()
    if RUNTIME_PARAMS['multiclass']:
        if RUNTIME_PARAMS['al']=='calf':
            mask[mask_original>0]=1
            mask[mask_original==7]=0 # right tibia marrow
            mask[mask_original==17]=0 # left tibia marrow
        else:
            mask[mask_original>0]=1
            mask[mask_original==11]=0 # right femur marrow
            mask[mask_original==31]=0 # left femur marrow
    
    if not np.array_equal(mask.shape,[target_size_y,target_size_x]):
        print('np.array_equal(mask.shape,[target_size_y,target_size_x]) is false',mask.shape,[target_size_y,target_size_x])
        assert(False)

    mask_sum=np.sum(mask)
    if (mask_sum==0): return MASK_VALIDITY_BLANKMASK

    if not np.array_equal(np.unique(mask),[0,1]):
        raise Exception("Mask values not 0 and 1: "+help_str)

    if mask_sum/np.prod(mask.shape)<0.03:
        if DEBUG: print('WARNING: %s with a value of %f, assuming this is not a valid mask'%(help_str,mask_sum/np.prod(mask.shape)))
        return MASK_VALIDITY_BAD

    QQ=np.where(mask==1)
    diffY=np.max(QQ[0])-np.min(QQ[0])
    assert diffY>0, 'diffY needs to be >0'
    ratio=float(diffY)/mask.shape[0]
    if ratio<0.5:
        if DEBUG: print('WARNING: ratio<0.5 for %s, assuming this is a one sided mask'%(help_str))
        return MASK_VALIDITY_SINGLESIDED
    
    return MASK_VALIDITY_VALID

def checkDixonImage(dixon_img):
    hist,_=np.histogram(dixon_img,bins=20)
    if np.argmax(hist)==0: return True
    return False
    
def load_BFC_image(filename,test):
        if 1 or not test or RUNTIME_PARAMS['al']=='calf' or simplerModel:
            if not os.path.exists(filename):
                raise Exception(f'ERROR: the following file does not exist {filename}')
            nibobj=nib.load(filename)
            return nibobj, nibobj.get_data()

        BFCfilename=filename.replace('.nii.gz','-bfc.nii.gz')
        if os.path.exists(BFCfilename):
            nibobj=nib.load(BFCfilename)
            return nibobj, nibobj.get_data()

        print('Bias correcting '+filename)
        os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/media/bkanber/WORK/mrtools/niftk-17.5.0/bin/ && /media/bkanber/WORK/mrtools/niftk-17.5.0/bin/niftkN4BiasFieldCorrection --sub 2 -i '%s' -o '%s'"%(filename,BFCfilename))

        nibobj=nib.load(BFCfilename)
        return nibobj, nibobj.get_data()
        
def load_case_base(inputdir,DIR,test=False):
    if DEBUG: print('load_case',DIR)
    TK=DIR.split('^')
    assert(len(TK)>=2)
    ll=TK[0]
    DIR=TK[1]

    if 'Amy_GOSH' in DIR:
        filename=glob.glob(os.path.join(DIR,'*DIXON_F.nii*'))[0]
    elif inputdir!='train' and inputdir!='validate':
        pattern=os.path.join(DIR,'fat.nii*')
        files=glob.glob(pattern)
        if len(files)<1:
            print(f'ERROR: No files found matching {pattern}')
            assert(False)
        filename=files[0]
    else:
        filename=os.path.join(DIR,'ana/fatfraction/'+ll+'/fat.nii.gz')
        if not os.path.exists(filename):
            filename=os.path.join(DIR,'ana/fatfraction/'+llshortdict[ll]+'/fat.nii.gz')

    if DEBUG: print('loading fat image')
    fatimgobj,fatimg=load_BFC_image(filename,test)

    numvox=np.product(fatimg.shape)
    if np.sum(fatimg>100)/numvox>=0.30: 
        fatimg*=0    
        raise Exception('Failed QC test '+DIR)

    if DEBUG: print('water image')
    if simplerModel:
        waterimg=fatimg*0
    else:
        if 'Amy_GOSH' in DIR:
            filename=glob.glob(os.path.join(DIR,'*DIXON_W.nii*'))[0]
        elif inputdir!='train' and inputdir!='validate':
            filename=glob.glob(os.path.join(DIR,'water.nii*'))[0]
        else:
            filename=os.path.join(DIR,'ana/fatfraction/'+ll+'/water.nii.gz')
            if not os.path.exists(filename):
                filename=os.path.join(DIR,'ana/fatfraction/'+llshortdict[ll]+'/water.nii.gz')

        waterimgobj,waterimg=load_BFC_image(filename,test)
        if not np.array_equal(fatimgobj.header.get_zooms(),waterimgobj.header.get_zooms()):
            raise Exception('Fat and water image resolutions are different for '+DIR)

    if DEBUG: print('loading DIXON 345ms image')
    if simplerModel:
        dixon_345img=fatimg*0
    else:
        if inputdir!='train' and inputdir!='validate':
            dixfile=glob.glob(os.path.join(DIR,'dixon345.nii*'))
        else:
            dixfile=glob.glob(os.path.join(DIR,'nii/*-Dixon_TE_345_'+ll+'.nii.gz')) # e.g. 1-0017-Dixon_TE_345_calf.nii.gz
            if len(dixfile)==0:
                dixfile=glob.glob(os.path.join(DIR,'nii/*-Dixon_TE_345_'+llshortdict[ll]+'.nii.gz')) # e.g. 1-0017-Dixon_TE_345_cf.nii.gz
            assert len(dixfile)==2, 'Failed len(dixfile)==2'

            id1=os.path.basename(dixfile[0]).replace('-Dixon_TE_345_'+ll+'.nii.gz','').replace('-Dixon_TE_345_'+llshortdict[ll]+'.nii.gz','').replace('-','.')
            id2=os.path.basename(dixfile[1]).replace('-Dixon_TE_345_'+ll+'.nii.gz','').replace('-Dixon_TE_345_'+llshortdict[ll]+'.nii.gz','').replace('-','.')
            try:
                id1=float(id1)
                id2=float(id2)
            except:
                id1=str(id1)
                id2=str(id2)
            assert(id1!=id2)

            if id1>id2: 
                dixfile=dixfile[::-1]

        dixon_345imgobj,dixon_345img=load_BFC_image(dixfile[0],test)
        assert checkDixonImage(dixon_345img), dixfile[0]+' may be a phase image'
        if not np.array_equal(fatimgobj.header.get_zooms(),dixon_345imgobj.header.get_zooms()):
            raise Exception('Fat and dixon_345 image resolutions are different for '+DIR)

    if simplerModel:
        dixon_460img=fatimg*0
    else:
        if inputdir!='train' and inputdir!='validate':
            dixfile=glob.glob(os.path.join(DIR,'dixon460.nii*'))
        else:
            dixfile=glob.glob(os.path.join(DIR,'nii/*-Dixon_TE_460_'+ll+'.nii.gz'))
            if len(dixfile)==0:
                dixfile=glob.glob(os.path.join(DIR,'nii/*-Dixon_TE_460_'+llshortdict[ll]+'.nii.gz')) 
            assert(len(dixfile)==2)

            id1=os.path.basename(dixfile[0]).replace('-Dixon_TE_460_'+ll+'.nii.gz','').replace('-Dixon_TE_460_'+llshortdict[ll]+'.nii.gz','').replace('-','.')
            id2=os.path.basename(dixfile[1]).replace('-Dixon_TE_460_'+ll+'.nii.gz','').replace('-Dixon_TE_460_'+llshortdict[ll]+'.nii.gz','').replace('-','.')
            try:
                id1=float(id1)
                id2=float(id2)
            except:
                id1=str(id1)
                id2=str(id2)
            assert(id1!=id2)

            if id1>id2: 
                dixfile=dixfile[::-1]

        dixon_460imgobj,dixon_460img=load_BFC_image(dixfile[0],test)
        #if not np.array_equal(fatimgobj.header.get_zooms(),dixon_460imgobj.header.get_zooms()):
        #    raise Exception('Fat and dixon_460 image resolutions are different for '+DIR)

        if 0 and dixfile[0]=='ibmcmt_p1/p1-010a/nii/0037-Dixon_TE_460_cf.nii.gz':
            pass
        else:
            assert checkDixonImage(dixon_460img), dixfile[0]+' may be a phase image'

        fatimg=dixon_460img

    if simplerModel:
        dixon_575img=fatimg*0
    else:
        if inputdir!='train' and inputdir!='validate':
            dixfile=glob.glob(os.path.join(DIR,'dixon575.nii*'))
        else:
            dixfile=glob.glob(os.path.join(DIR,'nii/*-Dixon_TE_575_'+ll+'.nii.gz'))
            if len(dixfile)==0:
                dixfile=glob.glob(os.path.join(DIR,'nii/*-Dixon_TE_575_'+llshortdict[ll]+'.nii.gz')) 
            assert(len(dixfile)==2)

            id1=os.path.basename(dixfile[0]).replace('-Dixon_TE_575_'+ll+'.nii.gz','').replace('-Dixon_TE_575_'+llshortdict[ll]+'.nii.gz','').replace('-','.')
            id2=os.path.basename(dixfile[1]).replace('-Dixon_TE_575_'+ll+'.nii.gz','').replace('-Dixon_TE_575_'+llshortdict[ll]+'.nii.gz','').replace('-','.')
            try:
                id1=float(id1)
                id2=float(id2)
            except:
                id1=str(id1)
                id2=str(id2)
            assert(id1!=id2)

            if id1>id2: 
                dixfile=dixfile[::-1]

        dixon_575imgobj,dixon_575img=load_BFC_image(dixfile[0],test)
        #if not np.array_equal(fatimgobj.header.get_zooms(),dixon_575imgobj.header.get_zooms()):
        #    raise Exception('Fat and dixon_575 image resolutions are different for '+DIR)

        if dixfile[0]=='ibmcmt_p1/p1-010a/nii/0037-Dixon_TE_575_cf.nii.gz':
            pass
        else:
            assert checkDixonImage(dixon_575img), dixfile[0]+' may be a phase image'

    # Mask selection (consider not using _af which are poor masks)
    if DEBUG: print('selecting mask')
    if 'brcalskd' in DIR:
        filename=os.path.join(DIR,'roi/Dixon345_'+llshortdict[ll]+'_uk_3.nii.gz')
    elif 'hypopp' in DIR:
        filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_bk.nii.gz')
        if not os.path.exists(filename):
            filename=os.path.join(DIR,'acq/ROI/'+llshortdict[ll]+'_dixon345_er_3.nii.gz')
        if not os.path.exists(filename):
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_er_3.nii.gz')
    elif 'ibmcmt_p1' in DIR:
        filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_bk.nii.gz')
        if not os.path.exists(filename):
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_jm_3.nii.gz')
        if not os.path.exists(filename):
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_gs_3.nii.gz')
        if not os.path.exists(filename):
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_me_3.nii.gz')
        if not os.path.exists(filename):
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_af_3.nii.gz')
    elif 'ibmcmt_p' in DIR:
        if ll=='calf' and DIR=='ibmcmt_p5/p5-027':
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_at_3.nii.gz')
        elif ll=='thigh' and DIR=='ibmcmt_p2/p2-008':
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_ta_3.nii.gz')
        elif ll=='calf' and DIR=='ibmcmt_p5/p5-044':
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_at_3.nii.gz')
        else:
            filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_bk.nii.gz')
            if not os.path.exists(filename):
                filename=os.path.join(DIR,'roi/Dixon345_'+llshortdict[ll]+'_bk.nii.gz')
            if not os.path.exists(filename):
                filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_me_3.nii.gz')
            if not os.path.exists(filename):
                filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_ta_3.nii.gz')
            if not os.path.exists(filename):
                filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_at_3.nii.gz')
            if not os.path.exists(filename):
                filename=os.path.join(DIR,'acq/ROI/'+ll+'_dixon345_af_3.nii.gz')
    else: filename=None

    #if not os.path.exists(filename):
    #    filename=None
    
    if DEBUG: print('loading mask')
    if filename is None:
        maskimg=np.zeros(fatimg.shape,dtype=np.uint8)
    else:
        maskimgobj=nib.load(filename)
        maskimg=maskimgobj.get_data()
        if not np.array_equal(fatimgobj.header.get_zooms(),maskimgobj.header.get_zooms()):
            raise Exception('Fat and mask image resolutions are different for '+DIR)

    if ll=='calf' and DIR=='ibmcmt_p2/p2-042': maskimg[:,:,3]=0
    if ll=='calf' and DIR=='ibmcmt_p3/p3-001': maskimg[:,:,2]=0
    if ll=='calf' and DIR=='brcalskd/BRCALSKD_011A': maskimg[:,:,5]=0
    if ll=='calf' and DIR=='ibmcmt_p5/p5-061': maskimg[:,:,4]=0
    if ll=='calf' and DIR=='ibmcmt_p5/p5-068': maskimg[:,:,5]=0
    if ll=='calf' and DIR=='brcalskd/BRCALSKD_002C': maskimg[:,:,6]=0

    if DEBUG: print('Converting/standardising mask')
    if not RUNTIME_PARAMS['multiclass']:
        if filename is not None and not convertMRCCentreMaskToBinary(DIR,ll,maskimg):
            raise Exception('convertMRCCentreMaskToBinary returned False')

        if ll=='calf' and DIR in ['brcalskd/BRCALSKD_028A','brcalskd/BRCALSKD_039C','hypopp/014_b','hypopp/006_a','ibmcmt_p5/p5-034','ibmcmt_p3/p3-008','ibmcmt_p6/p6-008']:
            pass
        elif ll=='thigh' and DIR in ['ibmcmt_p6/p6-008','hypopp/023_a','ibmcmt_p4/p4-027','ibmcmt_p4/p4-033','ibmcmt_p4/p4-062','ibmcmt_p4/p4-004','ibmcmt_p4/p4-046','brcalskd/BRCALSKD_028A','brcalskd/BRCALSKD_036C','ibmcmt_p2/p2-072','ibmcmt_p1/p1-014b','ibmcmt_p3/p3-067','ibmcmt_p3/p3-051','ibmcmt_p3/p3-011','ibmcmt_p2/p2-010','ibmcmt_p2/p2-041','ibmcmt_p4/p4-060']:
            pass
        elif (inputdir=='train' or inputdir=='validate') and not simplerModel:
            QQ=maskimg>0
            if np.sum(np.logical_and(fatimg[QQ]<10,waterimg[QQ]<10))>150:
                if False:
                    print('water',waterimg[QQ][np.logical_and(fatimg[QQ]<10,waterimg[QQ]<10)])
                    print('fat',fatimg[QQ][np.logical_and(fatimg[QQ]<10,waterimg[QQ]<10)])
                print('CHECK_VAL',np.sum(np.logical_and(fatimg[QQ]<10,waterimg[QQ]<10)))
                print(filename)
                raise Exception('Please check mask for '+DIR)
    else:
        if filename is not None and not convertMRCCentreMaskToStandard(DIR,ll,maskimg):
            raise Exception('convertMRCCentreMaskToStandard returned False')
        
    return (fatimg,waterimg,dixon_345img,dixon_575img,maskimg)

def load_case(inputdir,DIR,test=False):
    try:
        (fatimg,waterimg,dixon_345img,dixon_575img,maskimg)=load_case_base(inputdir,DIR,test)
    except Exception as e:
        print(repr(e))
        print('Could not get image data for '+DIR)
        return None

    assert(fatimg.shape==waterimg.shape)
    assert(fatimg.shape==dixon_345img.shape)
    assert(fatimg.shape==dixon_575img.shape)
    assert(fatimg.shape==maskimg.shape)

    t = type(maskimg[0,0,0])
    if t is not np.uint8 and t is not np.uint16:
        if DIR in ['calf^ibmcmt_p5/p5-034']:
            pass
        else:
            raise Exception('dtype not uint8/16 for mask '+DIR+' it is, '+str(t))
    
    maskimg=maskimg.astype(np.uint8)

    return DIR,fatimg,waterimg,dixon_345img,dixon_575img,maskimg

def load_data(DIRS, test=False):
    print('load_data() RUNTIME_PARAMS',RUNTIME_PARAMS)
    start_time = time.time()
    if len(DIRS)<1: 
        raise Exception('No data to load')

    print('Reading %d item(s)...'%len(DIRS))
    n_jobs=1 if MY_PC else max(1,multiprocessing.cpu_count()//2)

    ret = Parallel(n_jobs=n_jobs,verbose=2)(delayed(load_case)(RUNTIME_PARAMS['inputdir'],DIR,test) for DIR in DIRS)

    X_DIR, X_fatimg, X_waterimg, X_dixon_345img, X_dixon_575img, X_maskimg = zip(*ret)

    print('Read data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return list(filter(lambda x: x is not None,X_DIR)), list(filter(lambda x: x is not None,X_fatimg)), list(filter(lambda x: x is not None,X_waterimg)), list(filter(lambda x: x is not None,X_dixon_345img)), list(filter(lambda x: x is not None,X_dixon_575img)), list(filter(lambda x: x is not None,X_maskimg))

def scale_to_target(img):
    assert(len(img.shape)==2)

    if np.array_equal(img.shape,[target_size_y,target_size_x]):
        return img

    if not RUNTIME_PARAMS['multiclass']:
        return scale2D(img,target_size_y,target_size_x,order=3,mode='nearest',cval=0.0,prefilter=True)
    else:
        return scale2D(img,target_size_y,target_size_x,order=0,mode='nearest',cval=0.0,prefilter=True)
    
def scale_A_to_B(A,B):
    assert(len(A.shape)==2)
    assert(len(B.shape)==2)

    if np.array_equal(A.shape,B.shape):
        return A

    if not RUNTIME_PARAMS['multiclass']:
        return scale2D(A,B.shape[0],B.shape[1],order=3,mode='nearest',cval=0.0,prefilter=True)
    else:
        return scale2D(A,B.shape[0],B.shape[1],order=0,mode='nearest',cval=0.0,prefilter=True)
    
def read_and_normalize_data(DIRS, test=False):
    DIR,fatimg,waterimg,dixon_345img,dixon_575img,maskimg = load_data(DIRS, test)
    if len(DIR)<1:
        raise Exception('No data loaded')

    DIR_new,fatimg_new,waterimg_new,dixon_345img_new,dixon_575img_new,maskimg_new=[],[],[],[],[],[]
    for imgi in range(0,len(DIR)):
        if not RUNTIME_PARAMS['multiclass']:
            if test:
                assert(np.array_equal(np.unique(maskimg[imgi]),[0,1]) or np.array_equal(np.unique(maskimg[imgi]),[0])) 
            else:
                assert(np.array_equal(np.unique(maskimg[imgi]),[0,1]))
        else:
            valid_values=[0,1,2,3,4,5,6,7,11,12,13,14,15,16,17] if RUNTIME_PARAMS['al']=='calf' else [0,1,2,3,4,5,6,7,8,9,10,11,21,22,23,24,25,26,27,28,29,30,31]
            if test:
                assert(np.array_equal(np.unique(maskimg[imgi]),valid_values) or np.array_equal(np.unique(maskimg[imgi]),[0])) 
            else:
               if not np.array_equal(np.unique(maskimg[imgi]),valid_values):
                   print(f'EXCLUDE: np.unique(maskimg[imgi])={np.unique(maskimg[imgi])} for {DIRS[imgi]}')
                   continue
        
        for slice in range(0,fatimg[imgi].shape[2]):
            mask_validity=valid_mask(scale_to_target(maskimg[imgi][:,:,slice]),DIR[imgi]+'^slice'+str(slice))
            TO_ADD=False
            if mask_validity==MASK_VALIDITY_VALID:
                TO_ADD=True
            elif mask_validity==MASK_VALIDITY_SINGLESIDED and not RUNTIME_PARAMS['multiclass']:
                half_size=maskimg[imgi].shape[0]//2
                if valid_mask(scale_to_target(maskimg[imgi][:half_size,:,slice]),DIR[imgi]+'^slice'+str(slice)+'^side1')==MASK_VALIDITY_VALID: 
                    for img_it in (fatimg,waterimg,dixon_345img,dixon_575img,maskimg):
                        img_it[imgi][:,:,slice]=scale_A_to_B(img_it[imgi][:half_size,:,slice],img_it[imgi][:,:,slice])
                    TO_ADD=True
                elif valid_mask(scale_to_target(maskimg[imgi][half_size:,:,slice]),DIR[imgi]+'^slice'+str(slice)+'^side2')==MASK_VALIDITY_VALID: 
                    for img_it in (fatimg,waterimg,dixon_345img,dixon_575img,maskimg):
                        img_it[imgi][:,:,slice]=scale_A_to_B(img_it[imgi][half_size:,:,slice],img_it[imgi][:,:,slice])
                    TO_ADD=True
                else:
                    TO_ADD=test
            else: 
                TO_ADD=test

            if TO_ADD:
                fatslice=scale_to_target(fatimg[imgi][:,:,slice])
                waterslice=scale_to_target(waterimg[imgi][:,:,slice])
                maskslice=scale_to_target(maskimg[imgi][:,:,slice])
                dixon_345slice=scale_to_target(dixon_345img[imgi][:,:,slice])
                dixon_575slice=scale_to_target(dixon_575img[imgi][:,:,slice])

                if not RUNTIME_PARAMS['multiclass']:
                    if test:
                        assert(np.array_equal(np.unique(maskslice),[0,1]) or np.array_equal(np.unique(maskslice),[0])) 
                    else:
                        assert(np.array_equal(np.unique(maskslice),[0,1]))
                else:
                    valid_values=[0,1,2,3,4,5,6,7,11,12,13,14,15,16,17] if RUNTIME_PARAMS['al']=='calf' else [0,1,2,3,4,5,6,7,8,9,10,11,21,22,23,24,25,26,27,28,29,30,31]
                    if test:
                        if not np.array_equal(np.unique(maskslice),valid_values) and not np.array_equal(np.unique(maskslice),[0]): 
                           print(f'EXCLUDE: np.unique(maskslice)={np.unique(maskslice)} for {DIRS[imgi]}, slice={slice}')
                           continue
                    else:
                       if not np.array_equal(np.unique(maskslice),valid_values):
                           print(f'EXCLUDE: np.unique(maskslice)={np.unique(maskslice)} for {DIRS[imgi]}, slice={slice}')
                           continue

                fatimg_new.append(fatslice)
                waterimg_new.append(waterslice)
                dixon_345img_new.append(dixon_345slice)
                dixon_575img_new.append(dixon_575slice)
                maskimg_new.append(maskslice)
                DIR_text=DIR[imgi]+'^slice'+str(slice)
                DIR_new.append(DIR_text)

    if not test and len(DIR_new)!=len(DIR):
        print('INFO: len(DIR_new)!=len(DIR)',len(DIR_new),len(DIR))

    DIR=DIR_new

    fatimg = np.array(fatimg_new, dtype=np.float32)
    waterimg = np.array(waterimg_new, dtype=np.float32)
    dixon_345img = np.array(dixon_345img_new, dtype=np.float32)
    dixon_575img = np.array(dixon_575img_new, dtype=np.float32)
    maskimg = np.array(maskimg_new, dtype=np.uint8)

    del DIR_new,fatimg_new,waterimg_new,dixon_345img_new,dixon_575img_new,maskimg_new

    print('fatimg shape and type',fatimg.shape,fatimg.dtype) 
    print('waterimg shape and type',waterimg.shape,waterimg.dtype) 
    print('dixon_345img shape and type',dixon_345img.shape,dixon_345img.dtype) 
    print('dixon_575img shape and type',dixon_575img.shape,dixon_575img.dtype) 
    print('maskimg shape and type',maskimg.shape,maskimg.dtype) 

    fatimg = fatimg.reshape(fatimg.shape[0], fatimg.shape[1], fatimg.shape[2], 1)
    waterimg = waterimg.reshape(waterimg.shape[0], waterimg.shape[1], waterimg.shape[2], 1)
    dixon_345img = dixon_345img.reshape(dixon_345img.shape[0], dixon_345img.shape[1], dixon_345img.shape[2], 1)
    dixon_575img = dixon_575img.reshape(dixon_575img.shape[0], dixon_575img.shape[1], dixon_575img.shape[2], 1)
    maskimg = maskimg.reshape(maskimg.shape[0], maskimg.shape[1], maskimg.shape[2], 1)

    if simplerModel:
        binimg=fatimg.copy()
        for i in range(0,fatimg.shape[0]):
            thisfatimg=fatimg[i].copy()
            thisbinimg=binimg[i].copy()

            cutoff=np.nanmax(thisfatimg)/5
            thisbinimg[thisfatimg>cutoff]=1
            thisbinimg[thisfatimg<=cutoff]=0

            binimg[i]=thisbinimg
        data=np.concatenate((fatimg,binimg),axis=3)
    else:
        data=np.concatenate((fatimg,dixon_345img,dixon_575img),axis=3)

    print('Data shape:', data.shape)
    print('Lesion mask shape:', maskimg.shape)

    print('Mean data before normalisation: '+str(np.mean(data)))
    print('Std data before normalisation: '+str(np.std(data)))

    if True:
        training_means = np.nanmean(data, axis=(1,2))
        training_stds = np.nanstd(data, axis=(1,2))
        training_stds_to_divide_by = training_stds.copy()
        training_stds_to_divide_by[training_stds_to_divide_by==0] = 1.0

        print('Data means matrix shape: ',training_means.shape)

        for i in range(0,data.shape[0]):
            for j in range(0,data.shape[3]):
                data[i,:,:,j] = (data[i,:,:,j] - training_means[i,j]) / training_stds_to_divide_by[i,j]
    else:
        for i in range(0,data.shape[0]):
            for j in range(0,data.shape[3]):
                data[i,:,:,j] -= np.nanmin(data[i,:,:,j])
                data[i,:,:,j] /= np.nanmax(data[i,:,:,j])
                # also try: clip negatives

    print('Mean data after normalisation: '+str(np.mean(data)))
    print('Std data after normalisation: '+str(np.std(data)))

    return DIR,data,maskimg

def MYNET(input_size = [target_size_y,target_size_x,2 if simplerModel else 3]):
    if RUNTIME_PARAMS['multiclass']:
        activation='softmax'
        RUNTIME_PARAMS['classes']=17+1 if RUNTIME_PARAMS['al']=='calf' else 31+1
        loss=sm.losses.cce_dice_loss
        metrics=[sm.metrics.f1_score]
    else:
        activation='sigmoid'
        RUNTIME_PARAMS['classes']=1
        loss=sm.losses.bce_dice_loss
        metrics=[sm.metrics.f1_score]
    
    model = sm.Unet('resnet34', encoder_weights='imagenet', input_shape=input_size, classes=RUNTIME_PARAMS['classes'], activation=activation)

    opt = Adam(lr = RUNTIME_PARAMS['lr'], amsgrad=False)
    print(opt)
    model.compile(optimizer = opt, loss=loss, metrics=metrics)
    return model

def calc_dice(test_id,test_mask,preds):
    if type(test_id)==str:
        test_id=[test_id]
        test_mask=[test_mask]
        preds=[preds]
    
    assert(type(test_id)==list)

    DSCs=[]
    cutoffs=[]
    for case_i in range(0,len(test_id)):
        if MASK_VALIDITY_VALID!=valid_mask(np.squeeze(test_mask[case_i]),test_id[case_i]):
            if DEBUG: print('DSC: None for %s as it has no valid ground truth'%(test_id[case_i]))
            continue 

        cutoffspace=np.linspace(0,1,100)
        DSCspace=[]
        for i in range(0,len(cutoffspace)):
            binpreds=preds[case_i].copy()
            binpreds[np.where(preds[case_i]>cutoffspace[i])]=1
            binpreds[np.where(preds[case_i]<=cutoffspace[i])]=0
            DSCspace.append(numpy_dice_coefficient(test_mask[case_i], binpreds, smooth=1.))

        bestDSC=np.max(DSCspace)
        bestcutoff=cutoffspace[np.argmax(DSCspace)]
        if DEBUG: print('DSC: %f at cut-off %f for %s'%(bestDSC,bestcutoff,test_id[case_i]))
        DSCs.append(bestDSC)
        cutoffs.append(bestcutoff)

        calc_dice_file='calc_dice_%s_%s.csv'%(RUNTIME_PARAMS['inputdir'].replace('/','_'),RUNTIME_PARAMS['al'])
        if 'calc_dice_num_calls' not in RUNTIME_PARAMS:
            RUNTIME_PARAMS['calc_dice_num_calls']=0
            with open(calc_dice_file,'wt') as outfile:
                outfile.write('id,DSC,best_cutoff\n')

        RUNTIME_PARAMS['calc_dice_num_calls']+=1
        with open(calc_dice_file,'at') as outfile:
            outfile.write('%s,%f,%f\n'%(test_id[case_i],bestDSC,bestcutoff))

    if len(DSCs)>0: meanDSC=np.mean(DSCs)
    else: 
        meanDSC=None

    if len(test_id)>1 and DEBUG:
        print('meanDSC: %s'%(str(meanDSC)))

    return DSCs,cutoffs

def print_scores(data,data_mask,preds,std_preds,test_id):
    print(data.shape,data.dtype)
    print(data_mask.shape,data_mask.dtype)
    print(preds.shape,preds.dtype)
    print(len(test_id))

    print('Saving results')
    for i in range(0,preds.shape[0]):
        DIR=test_id[i]
        #DSCs,cutoffs=calc_dice(DIR,np.squeeze(data_mask[i]),preds[i])
        #assert(len(DSCs) in [0,1])

        #if len(DSCs)==1:
        #    results_DSC_file='results_DSC_%s_%s.csv'%(RUNTIME_PARAMS['inputdir'].replace('/','_'),RUNTIME_PARAMS['al'])
        #    if 'print_scores_num_calls' not in RUNTIME_PARAMS:
        #        RUNTIME_PARAMS['print_scores_num_calls']=0
        #        with open(results_DSC_file,'wt') as outfile:
        #            outfile.write('id,DSC,best_cutoff\n')

        #    RUNTIME_PARAMS['print_scores_num_calls']+=1
        #    with open(results_DSC_file,'at') as outfile:
        #        outfile.write('%s,%f,%f\n'%(DIR,DSCs[0],cutoffs[0]))

        TK=DIR.split('^')
        assert(len(TK)==3)
        ll=TK[0]
        DIR=TK[1]
        slice=int(TK[2].replace('slice',''))

        filename="%s/cnn-%s.nii.gz"%(DIR,ll)
        filename_std="%s/std-%s.nii.gz"%(DIR,ll)
        maskimg=None
        maskimg_std=None
        
        if slice>0:
            try:
                nibobj=nib.load(filename)
                maskimg=nibobj.get_data().astype(np.float32)
            except:
                print('Could not load '+filename)
            try:
                nibobj_std=nib.load(filename_std)
                maskimg_std=nibobj_std.get_data().astype(np.float32)
            except:
                print('Could not load '+filename_std)

        if maskimg is None or maskimg_std is None:
            if 'Amy_GOSH' in DIR:
                fatfilename=glob.glob(os.path.join(DIR,'*DIXON_F.nii*'))[0]
            elif RUNTIME_PARAMS['inputdir']!='train' and RUNTIME_PARAMS['inputdir']!='validate':
                fatfilename=glob.glob(os.path.join(DIR,'fat.nii*'))[0]
            else:
                fatfilename=os.path.join(DIR,'ana/fatfraction/'+ll+'/fat.nii.gz')
                if not os.path.exists(fatfilename):
                    fatfilename=os.path.join(DIR,'ana/fatfraction/'+llshortdict[ll]+'/fat.nii.gz')

            nibobj=nib.load(fatfilename)
            nibobj_std=nib.load(fatfilename)
            shape=nibobj.get_data().shape[0:3]
            maskimg=np.zeros(shape,dtype=np.float32)
            maskimg_std=np.zeros(shape,dtype=np.float32)

        assert(slice>=0)
        assert(slice<maskimg.shape[2])

        if RUNTIME_PARAMS['multiclass']:
            img_to_save=scale2D(np.argmax(preds[i],axis=2),maskimg.shape[0],maskimg.shape[1])
            std_img_to_save=scale2D(np.argmax(std_preds[i],axis=2),maskimg.shape[0],maskimg.shape[1])
        else:
            img_to_save=scale2D(preds[i],maskimg.shape[0],maskimg.shape[1])
            std_img_to_save=scale2D(std_preds[i],maskimg.shape[0],maskimg.shape[1])

        maskimg[:,:,slice]=img_to_save
        maskimg_std[:,:,slice]=std_img_to_save

        if DEBUG: print('Saving '+filename+' (slice %d)'%(slice))
        save_nifti(maskimg,nibobj.affine,nibobj.header,filename)

        if DEBUG: print('Saving '+filename_std+' (slice %d)'%(slice))
        save_nifti(maskimg_std,nibobj_std.affine,nibobj_std.header,filename_std)

RUNTIME_PARAMS['lr']=1E-3
RUNTIME_PARAMS['lr_stage']=1

#def MyLRscheduler(epoch, current_lr):
#    new_lr=float(RUNTIME_PARAMS['lr'])
#    print('\nStarting epoch %d, learning rate %.1e'%(epoch+1,new_lr))
#    return new_lr

class MyEarlyStopping(Callback):
    def __init__(self,model,bestweightsfile):
        self.model=model
        self.bestweightsfile=bestweightsfile
        
    def on_train_begin(self,logs):
        self.val_losses=[]

    def on_epoch_end(self, epoch, logs):
        assert(epoch==len(self.val_losses))

        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)

        min_loss_epoch=np.argmin(self.val_losses)
        
        if min_loss_epoch==len(self.val_losses)-1:
            print('Loss reduced to %.4f, saving weights'%(val_loss))
            self.model.save_weights(self.bestweightsfile,overwrite=True,save_format='h5')

        if epoch>=min_loss_epoch+10:
            print('Not improved for 10 epochs, stopping early')
            self.model.stop_training = True

class MyTrainingStatsCallback(Callback):
    def on_train_begin(self,logs):
        self.val_losses=[]

    def on_epoch_end(self, epoch, logs):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        min_loss_epoch=np.argmin(self.val_losses)
        print('Epoch %d, min loss %.4f at epoch %d'%(epoch+1,self.val_losses[min_loss_epoch],min_loss_epoch+1))

def MyGenerator(image_generator,mask_generator,data_gen_args): 
    while True:
        batch_images=image_generator.next() # (batch_size, 320, 160, 3)
        mask_images=mask_generator.next() 
        
        if data_gen_args['fill_mode']=='constant' and np.isnan(data_gen_args['cval']):
            mask_images=np.nan_to_num(mask_images,nan=0)
            for batch_i in range(0,batch_images.shape[0]):
                for ch_i in range(0,batch_images.shape[3]):
                    fill_value=np.random.randint(-5,+6) # TODO: optimise
                    batch_images[batch_i,:,:,ch_i]=np.nan_to_num(batch_images[batch_i,:,:,ch_i],
                        nan=fill_value)

        if RUNTIME_PARAMS['multiclass']:
            mask_images=tf.one_hot(mask_images.squeeze(axis=3),RUNTIME_PARAMS['classes'])

        yield batch_images, mask_images

def train(train_DIRS,test_DIRS,BREAK_OUT_AFTER_FIRST_FOLD):
    "Train and validate"
    test_DIR,test_data,test_maskimg = read_and_normalize_data(test_DIRS,True)
    train_DIR,train_data,train_maskimg = read_and_normalize_data(train_DIRS)
    
    if RUNTIME_PARAMS['multiclass']:
        print('Not validating mask values in multiclass mode')
    else:
        assert(np.array_equal(np.unique(train_maskimg),[0,1]))
        assert(np.array_equal(np.unique(test_maskimg),[0,1]) or np.array_equal(np.unique(test_maskimg),[0]))

    outer_train_subjects=[]
    for i in range(0,len(train_DIR)):
        outer_train_subjects.append(get_subject_id_from_DIR(train_DIR[i])) 

    outer_train_subjects=np.array(outer_train_subjects) 

    kf = GroupKFold(n_splits=5)
    fold=0
    for train_index, valid_index in kf.split(train_data,groups=outer_train_subjects):
        if RUNTIME_PARAMS['inputdir']=='train':
            print('fold',fold+1)
        else:
            print('outer fold',RUNTIME_PARAMS['outerfold']+1,'inner fold',fold+1)
    
        X_train_this,y_train_this=train_data[train_index],train_maskimg[train_index]
        X_valid_this,y_valid_this=train_data[valid_index],train_maskimg[valid_index]
        DIR_valid=list(np.array(train_DIR)[valid_index])

        print('X_train_this',X_train_this.shape,X_train_this.dtype)
        print('X_valid_this',X_valid_this.shape,X_valid_this.dtype)

        train_subjects=outer_train_subjects[train_index]
        valid_subjects=outer_train_subjects[valid_index]

        common_subjects=np.intersect1d(train_subjects, valid_subjects, assume_unique=False, return_indices=False)
        assert(common_subjects.size==0)

        print('batch_size: %d'%(batch_size))
        
        data_gen_args = dict(
#            rotation_range=10,
            width_shift_range=0.5,
            height_shift_range=0.5,
#            shear_range=10,
            zoom_range=[0.5,1.3] if RUNTIME_PARAMS['al']=='thigh' else [0.5,1.3],
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='constant', # wrap,reflect (nearest causes smudges)
            cval=np.nan,
        )

        #if True: data_gen_args = dict() # disable augmentation

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        seed = 1
        image_datagen.fit(X_train_this, augment=True, seed=seed)
        mask_datagen.fit(y_train_this, augment=True, seed=seed)

        image_generator = image_datagen.flow(
            X_train_this,
            batch_size=batch_size,
            seed=seed)

        mask_generator = mask_datagen.flow(
            y_train_this,
            batch_size=batch_size,
            seed=seed)

        steps_per_epoch=math.ceil(X_train_this.shape[0]/batch_size)*1

        retry=0
        while True:
            model=MYNET()
            model.summary()
            
            TEMP_WEIGHTS_FILE='tmp.'+''.join(random.choice(string.ascii_letters) for i in range(10))+'.weights.h5'

            callbacks = [
                MyEarlyStopping(model,TEMP_WEIGHTS_FILE),
                #LearningRateScheduler(MyLRscheduler),
                #TensorBoard(log_dir=INSTALL_DIR,histogram_freq=0,write_graph=False,write_images=False)
            ]

            if RUNTIME_PARAMS['multiclass']:
                validation_data=(X_valid_this, tf.one_hot(y_valid_this.squeeze(axis=3),RUNTIME_PARAMS['classes']))
            else:
                validation_data=(X_valid_this, y_valid_this.astype(np.float32))
            
            history=model.fit_generator(MyGenerator(image_generator,mask_generator,data_gen_args), 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=5000, 
                    shuffle=True, verbose=2, validation_data=validation_data, 
                    callbacks=callbacks)
            
            if RUNTIME_PARAMS['multiclass']: break
            
            if 'val_accuracy' in history.history:
                val_accuracy=history.history['val_accuracy']
            else:
                val_accuracy=history.history['val_acc']
                
            if val_accuracy[len(val_accuracy)-1]>0.90:
                break

            retry+=1
            print('Retrying training [retry=%d]'%(retry))
            os.remove(TEMP_WEIGHTS_FILE)

        model.load_weights(TEMP_WEIGHTS_FILE)

        if RUNTIME_PARAMS['inputdir']=='train':
            shutil.move(TEMP_WEIGHTS_FILE,'%s.%d.%s.%s.weights'%('simple' if simplerModel else 'full',fold,RUNTIME_PARAMS['al'],'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary'))
            trainingMetricsFilename='%s.%s.%d.%s.%s.pdf'%(RUNTIME_PARAMS['inputdir'].replace('/','-'),'simple' if simplerModel else 'full',fold,RUNTIME_PARAMS['al'],'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary') 
        else:
            os.remove(TEMP_WEIGHTS_FILE)
            trainingMetricsFilename='%s.%s.%d.%d.%s.%s.pdf'%(RUNTIME_PARAMS['inputdir'].replace('/','-'),'simple' if simplerModel else 'full',RUNTIME_PARAMS['outerfold'],fold,RUNTIME_PARAMS['al'],'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary') 

        if RUNTIME_PARAMS['multiclass']:
            pass
        else:
            saveTrainingMetrics(history,trainingMetricsFilename,trainingMetricsFilename)

            # validation
            p=model.predict(X_valid_this,batch_size=batch_size, verbose=1)
            DSCs,cutoffs=calc_dice(DIR_valid,y_valid_this,p)

            if fold==0:
                val_DSCs = DSCs
                val_cutoffs = cutoffs
                val_losses = [np.min(history.history['val_loss'])]
            else:
                val_DSCs.extend(DSCs)
                val_cutoffs.extend(cutoffs)
                val_losses.append(np.min(history.history['val_loss']))

            print('inner fold',fold+1)
            print('batch_size',batch_size)
            print('mean DSC [val] %.4f +- %.4f [n=%d]'%(np.mean(val_DSCs),np.std(val_DSCs),len(val_DSCs)))
            print('range DSC [val] %.4f - %.4f'%(np.min(val_DSCs),np.max(val_DSCs)))
            print('mean loss [val] %.4f +- %.4f'%(np.mean(val_losses),np.std(val_losses)))
            print('range loss [val] %.4f - %.4f'%(np.min(val_losses),np.max(val_losses)))
            print('mean best_cutoff %f'%(np.mean(val_cutoffs)))

        # test
        p=model.predict(test_data,batch_size=batch_size, verbose=1)
        
        if fold==0:
            preds = p
        else:
            preds = np.concatenate((preds,p),axis=3)
        fold+=1
        if BREAK_OUT_AFTER_FIRST_FOLD:
            break

    mean_preds=np.nanmean(preds,axis=3)
    std_preds=np.nanstd(preds,axis=3)
    DSCs,_cutoffs=calc_dice(test_DIR,test_maskimg,mean_preds)
    print_scores(test_data,test_maskimg,mean_preds,std_preds,test_DIR)
    return DSCs

def test(test_DIRS):
    print('test',test_DIRS)
    print('RUNTIME_PARAMS',RUNTIME_PARAMS)
    test_DIR,test_data,test_maskimg = read_and_normalize_data(test_DIRS,True)    
    assert(np.array_equal(np.unique(test_maskimg),[0,1]) or np.array_equal(np.unique(test_maskimg),[0]))

    model=MYNET()
    for fold in range(0,5):
        print('batch_size: %d'%(batch_size))

        weightsfile='%s.%d.%s.%s.weights'%('simple' if simplerModel else 'full',fold,RUNTIME_PARAMS['al'],'multiclass' if RUNTIME_PARAMS['multiclass'] else 'binary')
        weightsfile=os.path.join(INSTALL_DIR,weightsfile)

        if not os.path.exists(weightsfile):
            continue
            msg='Downloading '+os.path.basename(weightsfile)
            print(msg)
            if RUNTIME_PARAMS['widget'] is not None:
                RUNTIME_PARAMS['widget']['text']=msg
                RUNTIME_PARAMS['widget'].update()
            url="https://github.com/bariskanber/musclesenseworkbench/releases/download/r1/%s"%(os.path.basename(weightsfile))
            urllib.request.urlretrieve(url, weightsfile)
        
        if RUNTIME_PARAMS['widget'] is not None:
            RUNTIME_PARAMS['widget']['text']='Calculating mask (%.0f%%)...'%(100*float(fold+1)/5)
            RUNTIME_PARAMS['widget'].update()

        model.load_weights(weightsfile)

        p=model.predict(test_data,batch_size=batch_size, verbose=1)
        p=np.expand_dims(p,axis=0)
        
        if fold==0:
            preds = p
        else:
            preds = np.concatenate((preds,p),axis=0)

    mean_preds=np.nanmean(preds,axis=0)
    std_preds=np.nanstd(preds,axis=0)

    #mean_preds-=std_preds*1.96
    
#    DSCs,_cutoffs=calc_dice(test_DIR,test_maskimg,mean_preds)
    print_scores(test_data,test_maskimg,mean_preds,std_preds,test_DIR)
    return DSCs

def main(al,inputdir,widget):
    RUNTIME_PARAMS['al']=al
    RUNTIME_PARAMS['inputdir']=inputdir
    RUNTIME_PARAMS['widget']=widget
    RUNTIME_PARAMS['multiclass']=True # individual muscle segmentation (vs. whole muscle)

    if RUNTIME_PARAMS['widget'] is not None:
        RUNTIME_PARAMS['widget']['text']='Calculating mask...'
        RUNTIME_PARAMS['widget'].update()

    ll=RUNTIME_PARAMS['al']

    start_time = time.time()

    if RUNTIME_PARAMS['inputdir']=='train' or RUNTIME_PARAMS['inputdir']=='validate':
        DIRS=[]
        for DATA_DIR in ['brcalskd','hypopp','ibmcmt_p1','ibmcmt_p2','ibmcmt_p3','ibmcmt_p4','ibmcmt_p5','ibmcmt_p6']:
                for de in glob.glob(os.path.join(DATA_DIR,'*')):
                    if not os.path.isdir(de):
                        if DEBUG: print(de+' is not a directory')
                        continue

                    if not os.path.isdir(os.path.join(de,'ana/fatfraction/'+ll)) and not os.path.isdir(os.path.join(de,'ana/fatfraction/'+llshortdict[ll])):
                        if DEBUG: print(de+' does not have a '+ll+' directory')
                        continue

                    files=glob.glob(os.path.join(de,'roi/?ixon345_'+ll+'_*.nii.gz'))
                    if len(files)==0:
                        files=glob.glob(os.path.join(de,'roi/?ixon345_'+llshortdict[ll]+'_*.nii.gz'))
                    if len(files)==0:
                        files=glob.glob(os.path.join(de,'acq/ROI/'+ll+'_*.nii.gz'))
                    if len(files)==0:
                        files=glob.glob(os.path.join(de,'acq/ROI/'+llshortdict[ll]+'_*.nii.gz'))
                    if len(files)==0:
                        if DEBUG: print(de+' does not have a '+ll+' mask')
                        continue

                    files=glob.glob(os.path.join(de,'nii/*_lg_*.nii.gz'))
                    for f in files:
                        os.rename(f,f.replace('_lg_','_'))

                    if False and ll=='calf' and de in [
                        'hypopp/015_b','hypopp/016_b','hypopp/017_b','hypopp/019_b','hypopp/022_b','ibmcmt_p2/p2-046','ibmcmt_p2/p2-026','ibmcmt_p4/p4-008','ibmcmt_p4/p4-029','ibmcmt_p4/p4-030','ibmcmt_p3/p3-030','ibmcmt_p4/p4-054','ibmcmt_p4/p4-071','ibmcmt_p5/p5-033','ibmcmt_p3/p3-046','ibmcmt_p5/p5-042','ibmcmt_p5/p5-044','ibmcmt_p5/p5-049','ibmcmt_p5/p5-060','ibmcmt_p6/p6-008','ibmcmt_p5/p5-032','ibmcmt_p6/p6-032','ibmcmt_p5/p5-039','ibmcmt_p6/p6-044','ibmcmt_p6/p6-062','ibmcmt_p4/p4-046','ibmcmt_p4/p4-049']:
                        if DEBUG: print('skipping %s that has L/R segmented on different slices or only one side segmented'%(de))
                        continue 

                    if ll=='thigh' and de in [
                        'ibmcmt_p3/p3-072', # mask is out-of-alignment
                        'ibmcmt_p5/p5-004', # looks a bit off
                        ]: 
                        if DEBUG: print('skipping %s that has bad mask'%(de))
                        continue

                    if False and ll=='calf' and de in [
                        'ibmcmt_p1/p1-001b', # looks quite badly segmented
                        ]: 
                        if DEBUG: print('skipping %s that has bad mask'%(de))
                        continue

                    if False and ll=='calf' and de in [
                        'ibmcmt_p2/p2-065', # actually might be good to include
                        'ibmcmt_p3/p3-047', # actually looks good
                        'ibmcmt_p3/p3-055', # actually looks good
                        'ibmcmt_p3/p3-008', # very noisy but add (at least to train)
                        'ibmcmt_p5/p5-034', # very bad but add (at least to train)
                        ]:
                        if DEBUG: print('skipping %s that have poor image quality'%(de))
                        continue

                    DIRS.append(ll+'^'+de)

                    #if MY_PC and len(DIRS)>20: break

        #DIRS=DIRS[:20]
        print('%d cases found'%(len(DIRS)))

        if RUNTIME_PARAMS['inputdir']=='train':
                train_DIRS=DIRS[:]
                if RUNTIME_PARAMS['al']=='calf':   
                    test_DIRS=[
                        'calf^ibmcmt_p2/p2-008',
                        ]
                else:
                    test_DIRS=[
                        'thigh^ibmcmt_p1/p1-007b',
                        ]

                train_DIRS=np.array(train_DIRS)
                test_DIRS=np.array(test_DIRS)

                np.random.shuffle(train_DIRS)

                train(train_DIRS,test_DIRS,BREAK_OUT_AFTER_FIRST_FOLD=True)
        elif RUNTIME_PARAMS['inputdir']=='validate':
            print('Validate mode')
            print('Use this mode to perform nested cross validation over all the available training data')
            print('')
            outloop_req=input('Input outer loop number (default=all): ').strip()
            if len(outloop_req)==0: outloop_req='all'
            print(outloop_req)

            difficult_cases=[
        #        'thigh^ibmcmt_p1/p1-007a',
        #        'thigh^ibmcmt_p1/p1-007b',
        #        'thigh^ibmcmt_p2/p2-008',
        #        'thigh^ibmcmt_p2/p2-008b',
        #        'calf^ibmcmt_p4/p4-044',
        #        'calf^ibmcmt_p4/p4-061',
        #        'calf^hypopp/006_b',
        #        'calf^ibmcmt_p2/p2-008',
        #        'calf^ibmcmt_p2/p2-030',
        #        'calf^ibmcmt_p2/p2-030b',
        #        'calf^ibmcmt_p2/p2-008b',
                ]

            DSCs=[]

            case_i=0
            for difficult_case in difficult_cases:
                case_i+=1
                print('Doing difficult case %d of %d with LOO'%(case_i,len(difficult_cases)))

                train_DIRS=DIRS[:]
                test_DIRS=[difficult_case]

                if difficult_case in train_DIRS: 
                    train_DIRS.remove(difficult_case)

                train_DIRS=np.array(train_DIRS)
                test_DIRS=np.array(test_DIRS)

                np.random.shuffle(train_DIRS)

                DSCarray=train(train_DIRS,test_DIRS,BREAK_OUT_AFTER_FIRST_FOLD=False)
                DSCs.extend(DSCarray)

                print('batch_size',batch_size)
                print('difficult case %d or %d'%(case_i,len(difficult_cases)))
                if len(DSCs)>0:
                    print('mean DSC %.4f +- %.4f [n=%d]'%(np.mean(DSCs),np.std(DSCs),len(DSCs)))
                    print('range DSC %.4f - %.4f'%(np.min(DSCs),np.max(DSCs)))

            if True:
                DIRS=np.array(DIRS) 

                np.random.shuffle(DIRS) 

                subjects=[]
                for i in range(0,len(DIRS)):
                    subjects.append(get_subject_id_from_DIR(DIRS[i]))

                subjects=np.array(subjects)   

                kf = GroupKFold(n_splits=5)
                fold=0

                for train_index, test_index in kf.split(DIRS,groups=subjects):
                    print('outer fold',fold+1)
                    RUNTIME_PARAMS['outerfold']=fold

                    if outloop_req!='all' and fold+1!=int(outloop_req): 
                        fold+=1
                        continue

                    train_DIRS=DIRS[train_index]
                    test_DIRS=DIRS[test_index]

                    train_subjects=subjects[train_index]
                    test_subjects=subjects[test_index]

                    common_subjects=np.intersect1d(train_subjects, test_subjects, assume_unique=False, return_indices=False)
                    assert(common_subjects.size==0)

                    print('Removing already tested cases from test set')
                    train_DIRS=list(train_DIRS)
                    test_DIRS=list(test_DIRS)
                    
                    for DIR in difficult_cases:
                        if DIR in test_DIRS: 
                            test_DIRS.remove(DIR)
                            train_DIRS.append(DIR)
                            
                    train_DIRS=np.array(train_DIRS)
                    test_DIRS=np.array(test_DIRS)

                    np.random.shuffle(train_DIRS)

                    DSCarray=train(train_DIRS,test_DIRS,
                        BREAK_OUT_AFTER_FIRST_FOLD=True)
                    DSCs.extend(DSCarray)

                    print('batch_size',batch_size)
                    print('outer fold',fold+1)
                    print('mean DSC %.4f +- %.4f [n=%d]'%(np.mean(DSCs),np.std(DSCs),len(DSCs)))
                    print('range DSC %.4f - %.4f'%(np.min(DSCs),np.max(DSCs)))

                    fold+=1
    else:
        DIRS=[]
        for DATA_DIR in [RUNTIME_PARAMS['inputdir']]:
                for de in glob.glob(os.path.join(DATA_DIR,'*')):
                    if not os.path.isdir(de):
                        if DEBUG: print(de+' is not a directory')
                        continue

                    DIRS.append(ll+'^'+de)

        if len(DIRS)==0:
            DIRS.append(ll+'^'+RUNTIME_PARAMS['inputdir'])

        print(DIRS)
        print('%d cases found'%(len(DIRS)))

        DIRS=np.array(DIRS)
        DSCarray=test(DIRS)
        
    print('Running time: {} hours'.format(round((time.time() - start_time)/3600.0, 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("mmseg_ll")
    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('-al', type=str, help='anatomical location (calf/thigh)')
    parser.add_argument('-inputdir', type=str, help='input directory/folder (or train/validate)')
    args=parser.parse_args()

    if args.inputdir is None or args.al is None:
        parser.print_help()
        sys.exit(1)

    main(args.al,args.inputdir,widget=None)
