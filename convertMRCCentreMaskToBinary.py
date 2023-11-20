import numpy as np

def convertMRCCentreMaskToBinary(DIR,ll,maskimg):
    if ll=='calf' and 'brcalskd' in DIR:
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=16)

        expected_vals=[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15] # Left/right tibial nerve not labelled
        actual_vals=np.unique(maskimg)
        if DIR.replace('data/','') not in ['brcalskd/BRCALSKD_013A','brcalskd/BRCALSKD_019A','brcalskd/BRCALSKD_023C','brcalskd/BRCALSKD_054A'] and not np.array_equal(actual_vals,expected_vals):
            raise Exception('Expected unique mask values were %s but are %s'%(expected_vals,actual_vals))

        maskimg[np.where(maskimg==7)]=0 # Right Tibia Marrow
        maskimg[np.where(maskimg==8)]=0 # Right tibial nerve
        maskimg[np.where(maskimg==15)]=0 # Left Tibia Marrow
        maskimg[np.where(maskimg==16)]=0 # Left tibial nerve

        maskimg[np.where(maskimg>0)]=1
    elif ll=='thigh' and 'brcalskd' in DIR:
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=23)

        maskimg[np.where(maskimg==11)]=0 # Right Femur marrow
        maskimg[np.where(maskimg==12)]=0 # Left Sciatic Nerve
        maskimg[np.where(maskimg==23)]=0 # Left Femur marrow

        maskimg[np.where(maskimg>0)]=1
    elif ll=='thigh' and 'ibmcmt_p' in DIR:
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=32)

        maskimg[np.where(maskimg==11)]=0 # Right Femur marrow
        maskimg[np.where(maskimg==12)]=0 # Right Sciatic Nerve
        maskimg[np.where(maskimg==31)]=0 # Left Femur marrow
        maskimg[np.where(maskimg==32)]=0 # Left Sciatic Nerve

        maskimg[np.where(maskimg>0)]=1
    elif ll=='thigh' and 'hypopp' in DIR:
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=32)

        maskimg[np.where(maskimg==11)]=0 # Right Femur marrow
        maskimg[np.where(maskimg==12)]=0 # 
        maskimg[np.where(maskimg==13)]=0 # 
        maskimg[np.where(maskimg==31)]=0 # Left Femur marrow
        maskimg[np.where(maskimg==32)]=0 #

        maskimg[np.where(maskimg>0)]=1
    elif ll=='calf' and ('hypopp' in DIR or 'ibmcmt_p1' in DIR or 'ibmcmt_p2' in DIR or 
             'ibmcmt_p3' in DIR or 'ibmcmt_p4' in DIR or 'ibmcmt_p5' in DIR or 
             'ibmcmt_p6' in DIR):
        assert(np.min(maskimg)==0)

        if DIR.replace('data/','') in ('hypopp/019_a','hypopp/012_b','ibmcmt_p2/p2-042','ibmcmt_p2/p2-018',
            'ibmcmt_p3/p3-044','ibmcmt_p3/p3-050','ibmcmt_p2/p2-030b','ibmcmt_p2/p2-008b'):
                maskimg[maskimg>18]=0 # These cases have suporous values>18 but masks otherwise look ok

        assert np.max(maskimg)<=18, 'ERROR: np.max(maskimg)>18 for '+DIR

        maskimg[np.where(maskimg==7)]=0 # Right Tibia Marrow
        maskimg[np.where(maskimg==8)]=0 # Right Subcut Fat Post/Right tibial nerve
        maskimg[np.where(maskimg==9)]=0 # Right Saline Bag
        maskimg[np.where(maskimg==17)]=0 # Left Tibia Marrow
        maskimg[np.where(maskimg==18)]=0 # Left Subcut Fat Post/Left tibial nerve
       
        maskimg[np.where(maskimg>0)]=1
    else:
        return False

    return True