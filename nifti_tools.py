import numpy as np
import nibabel as nib

debug=False

def save_nifti(data,affine=np.eye(4),header=None,filename='save_nifti.nii.gz',zooms=None):
    if header is not None:
        # check/update cal_min/max
        cal_min=np.min(data)
        cal_max=np.max(data)
        if 'cal_min' not in header or 'cal_max' not in header or header['cal_min']!=cal_min or header['cal_max']!=cal_max:
            header['cal_min']=cal_min
            header['cal_max']=cal_max
            if debug:
                print('Adjusted cal_min/max for '+filename)

        nibobj=nib.Nifti1Image(data,affine,header)
    else:
        nibobj=nib.Nifti1Image(data,affine)

    if zooms is not None:
        nibobj.header.set_zooms(zooms) 

    nib.save(nibobj,filename)

