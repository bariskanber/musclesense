import numpy as np

def convertMRCCentreMaskToStandard(DIR,ll,maskimg):
    maskimg_original=maskimg.copy()
    
    if len(np.unique(maskimg))<3:
        raise Exception('Not enough labels detected, len(np.unique(maskimg)) = '+str(len(np.unique(maskimg))))
    
    if ll=='hand':
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=23)

        expected_vals=[0,19,20,22,23]
        actual_vals=np.unique(maskimg)
        if not np.array_equal(actual_vals,expected_vals):
            raise Exception('Expected unique mask values were %s but are %s'%(expected_vals,actual_vals))

        maskimg[np.where(maskimg_original==0)]=0
        maskimg[np.where(maskimg_original==19)]=19
        maskimg[np.where(maskimg_original==20)]=20
        maskimg[np.where(maskimg_original==22)]=22
        maskimg[np.where(maskimg_original==23)]=23
    elif ll=='calf' and ('brcalskd' in DIR or 'alscs' in DIR):
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=16)

        expected_vals=[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15] # Left/right tibial nerve not labelled
        actual_vals=np.unique(maskimg)
        if DIR.replace('data/','') not in ['brcalskd/BRCALSKD_013A','brcalskd/BRCALSKD_019A','brcalskd/BRCALSKD_023C','brcalskd/BRCALSKD_054A'] and not np.array_equal(actual_vals,expected_vals):
            raise Exception('Expected unique mask values were %s but are %s'%(expected_vals,actual_vals))

        maskimg[np.where(maskimg_original==0)]=0
        maskimg[np.where(maskimg_original==1)]=1
        maskimg[np.where(maskimg_original==2)]=2
        maskimg[np.where(maskimg_original==3)]=3
        maskimg[np.where(maskimg_original==4)]=4
        maskimg[np.where(maskimg_original==5)]=5
        maskimg[np.where(maskimg_original==6)]=6
        maskimg[np.where(maskimg_original==7)]=7
        maskimg[np.where(maskimg_original==8)]=0
        maskimg[np.where(maskimg_original==9)]=11
        maskimg[np.where(maskimg_original==10)]=12
        maskimg[np.where(maskimg_original==11)]=13
        maskimg[np.where(maskimg_original==12)]=14
        maskimg[np.where(maskimg_original==13)]=15
        maskimg[np.where(maskimg_original==14)]=16
        maskimg[np.where(maskimg_original==15)]=17
        maskimg[np.where(maskimg_original==16)]=0
    elif ll=='thigh' and ('brcalskd' in DIR or 'alscs' in DIR):
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=23)

        maskimg[np.where(maskimg_original==0)]=0
        maskimg[np.where(maskimg_original==1)]=1
        maskimg[np.where(maskimg_original==2)]=2
        maskimg[np.where(maskimg_original==3)]=3
        maskimg[np.where(maskimg_original==4)]=4
        maskimg[np.where(maskimg_original==5)]=5
        maskimg[np.where(maskimg_original==6)]=6
        maskimg[np.where(maskimg_original==7)]=7
        maskimg[np.where(maskimg_original==8)]=8
        maskimg[np.where(maskimg_original==9)]=9
        maskimg[np.where(maskimg_original==10)]=10
        maskimg[np.where(maskimg_original==11)]=11
        maskimg[np.where(maskimg_original==12)]=0
        maskimg[np.where(maskimg_original==13)]=21
        maskimg[np.where(maskimg_original==14)]=22
        maskimg[np.where(maskimg_original==15)]=23
        maskimg[np.where(maskimg_original==16)]=24
        maskimg[np.where(maskimg_original==17)]=25
        maskimg[np.where(maskimg_original==18)]=26
        maskimg[np.where(maskimg_original==19)]=27
        maskimg[np.where(maskimg_original==20)]=28
        maskimg[np.where(maskimg_original==21)]=29
        maskimg[np.where(maskimg_original==22)]=30
        maskimg[np.where(maskimg_original==23)]=31
    elif ll=='thigh' and ('poems' in DIR or 'ibmcmt_p' in DIR or 'arimoclomol' in DIR or 'mdacmt' in DIR or 'dhmn' in DIR):    
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=33)

        maskimg[np.where(maskimg_original==0)]=0
        maskimg[np.where(maskimg_original==1)]=1
        maskimg[np.where(maskimg_original==2)]=2
        maskimg[np.where(maskimg_original==3)]=3
        maskimg[np.where(maskimg_original==4)]=4
        maskimg[np.where(maskimg_original==5)]=5
        maskimg[np.where(maskimg_original==6)]=6
        maskimg[np.where(maskimg_original==7)]=7
        maskimg[np.where(maskimg_original==8)]=8
        maskimg[np.where(maskimg_original==9)]=9
        maskimg[np.where(maskimg_original==10)]=10
        maskimg[np.where(maskimg_original==11)]=11
        maskimg[np.where(maskimg_original==12)]=0
        maskimg[np.where(maskimg_original==13)]=0
        maskimg[np.where(maskimg_original==21)]=21
        maskimg[np.where(maskimg_original==22)]=22
        maskimg[np.where(maskimg_original==23)]=23
        maskimg[np.where(maskimg_original==24)]=24
        maskimg[np.where(maskimg_original==25)]=25
        maskimg[np.where(maskimg_original==26)]=26
        maskimg[np.where(maskimg_original==27)]=27
        maskimg[np.where(maskimg_original==28)]=28
        maskimg[np.where(maskimg_original==29)]=29
        maskimg[np.where(maskimg_original==30)]=30
        maskimg[np.where(maskimg_original==31)]=31
        maskimg[np.where(maskimg_original==32)]=0
        maskimg[np.where(maskimg_original==33)]=0
    elif ll=='thigh' and 'hypopp' in DIR:
        assert(np.min(maskimg)==0)
        assert(np.max(maskimg)<=32)

        maskimg[np.where(maskimg_original==0)]=0
        maskimg[np.where(maskimg_original==1)]=1
        maskimg[np.where(maskimg_original==2)]=2
        maskimg[np.where(maskimg_original==3)]=3
        maskimg[np.where(maskimg_original==4)]=4
        maskimg[np.where(maskimg_original==5)]=5
        maskimg[np.where(maskimg_original==6)]=6
        maskimg[np.where(maskimg_original==7)]=7
        maskimg[np.where(maskimg_original==8)]=8
        maskimg[np.where(maskimg_original==9)]=9
        maskimg[np.where(maskimg_original==10)]=10
        maskimg[np.where(maskimg_original==11)]=11
        maskimg[np.where(maskimg_original==12)]=0
        maskimg[np.where(maskimg_original==13)]=0
        maskimg[np.where(maskimg_original==21)]=21
        maskimg[np.where(maskimg_original==22)]=22
        maskimg[np.where(maskimg_original==23)]=23
        maskimg[np.where(maskimg_original==24)]=24
        maskimg[np.where(maskimg_original==25)]=25
        maskimg[np.where(maskimg_original==26)]=26
        maskimg[np.where(maskimg_original==27)]=27
        maskimg[np.where(maskimg_original==28)]=28
        maskimg[np.where(maskimg_original==29)]=29
        maskimg[np.where(maskimg_original==30)]=30
        maskimg[np.where(maskimg_original==31)]=31
        maskimg[np.where(maskimg_original==32)]=0
    elif ll=='calf' and ('poems' in DIR or 'hypopp' in DIR or 'ibmcmt_p' in DIR or 'mdacmt' in DIR or 'dhmn' in DIR):
        assert(np.min(maskimg)==0)

        if DIR.replace('data/','') in ['hypopp/019_a','hypopp/012_b','ibmcmt_p2/p2-042','ibmcmt_p2/p2-018','ibmcmt_p3/p3-044','ibmcmt_p3/p3-050','ibmcmt_p2/p2-030b','ibmcmt_p2/p2-008b']:
            maskimg[maskimg>18]=0 # These cases have suporous values>18 but masks otherwise look ok
        
        assert np.max(maskimg)<=19, 'ERROR: np.max(maskimg)>19 for '+DIR

        maskimg[np.where(maskimg_original==0)]=0
        maskimg[np.where(maskimg_original==1)]=1
        maskimg[np.where(maskimg_original==2)]=2
        maskimg[np.where(maskimg_original==3)]=3
        maskimg[np.where(maskimg_original==4)]=4
        maskimg[np.where(maskimg_original==5)]=5
        maskimg[np.where(maskimg_original==6)]=6
        maskimg[np.where(maskimg_original==7)]=7
        maskimg[np.where(maskimg_original==8)]=0
        maskimg[np.where(maskimg_original==9)]=0
        maskimg[np.where(maskimg_original==11)]=11
        maskimg[np.where(maskimg_original==12)]=12
        maskimg[np.where(maskimg_original==13)]=13
        maskimg[np.where(maskimg_original==14)]=14
        maskimg[np.where(maskimg_original==15)]=15
        maskimg[np.where(maskimg_original==16)]=16
        maskimg[np.where(maskimg_original==17)]=17
        maskimg[np.where(maskimg_original==18)]=0
        maskimg[np.where(maskimg_original==19)]=0
    else:
        return False

    return True
