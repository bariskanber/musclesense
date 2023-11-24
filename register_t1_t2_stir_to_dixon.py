import os
import glob

def register_t1_t2_stir_to_dixon(filename: str, ll: str, llshortdict: dict):
    if filename.startswith('data/ibmcmt_p1/'):
        t1t2stir = 'data/ibmcmt_t1_t2_stir/' + filename.replace('data/ibmcmt_p1/','')
    elif filename.startswith('data/ibmcmt_p2/'):
        t1t2stir = 'data/ibmcmt_t1_t2_stir/' + filename.replace('data/ibmcmt_p2/','')
    elif filename.startswith('data/ibmcmt_p3/'):
        t1t2stir = 'data/ibmcmt_t1_t2_stir/' + filename.replace('data/ibmcmt_p3/','')
    elif filename.startswith('data/ibmcmt_p4/'):
        t1t2stir = 'data/ibmcmt_t1_t2_stir/' + filename.replace('data/ibmcmt_p4/','')
    elif filename.startswith('data/ibmcmt_p5/'):
        t1t2stir = 'data/ibmcmt_t1_t2_stir/' + filename.replace('data/ibmcmt_p5/','')
    elif filename.startswith('data/ibmcmt_p6/'):
        t1t2stir = 'data/ibmcmt_t1_t2_stir/' + filename.replace('data/ibmcmt_p6/','')
    elif filename.startswith('data/brcalskd/'):
        t1t2stir = 'data/brcalskd_t1_t2_stir/' + filename.replace('data/brcalskd/','')
    elif filename.startswith('data/hypopp/'):
        t1t2stir = 'data/hypopp_t1_t2_stir/' + filename.replace('data/hypopp/','')
    else:
        raise Exception('Unexpected condition')
    
    t1t2stir = os.path.dirname(t1t2stir)
    if not os.path.isdir(t1t2stir):
        raise Exception('T1T2STIR dir %s not found'%(t1t2stir))
    
    #keywords = ['stir','STIR']
    keywords = ['t1w','t1_tse']
    
    for keyword in keywords:
        dirlist = glob.glob('%s/*%s*_%s.nii.gz'%(t1t2stir,keyword,ll))
        if len(dirlist)==0:
            dirlist = glob.glob('%s/*%s*_%s.nii.gz'%(t1t2stir,keyword,llshortdict[ll]))
        else:
            break
    
    if len(dirlist)==0:
        if 'stir' in keywords and ll=='calf' and t1t2stir in ['data/brcalskd_t1_t2_stir/BRCALSKD_003A/nii',
                        'data/brcalskd_t1_t2_stir/BRCALSKD_002A/nii',
                        'data/ibmcmt_t1_t2_stir/p1-011b/nii',
                        'data/ibmcmt_t1_t2_stir/p2-042/nii',
                        ]:
            pass
        elif 'stir' in keywords and ll=='thigh' and t1t2stir in ['data/brcalskd_t1_t2_stir/BRCALSKD_003A/nii',
                        'data/brcalskd_t1_t2_stir/BRCALSKD_002A/nii',
                        ]:
            pass
        else:
            raise Exception('No %s found in %s'%(keywords,t1t2stir))
        
    for toreg in dirlist:
        print('Register '+toreg+' to '+filename)
        outputnii = os.path.join(os.path.dirname(filename),'%s_%s_dixon_space_%s'%(keywords[0],ll,os.path.basename(toreg)))
        print('outputnii',outputnii)
        if os.path.exists(outputnii):
            print('Aready done')
            continue
        cmd = 'reg_aladin -ref %s -flo %s -aff %s -res %s -omp 4'%(filename, toreg ,outputnii.replace('.nii.gz','_affine.txt'), outputnii)
        print(cmd)
        import subprocess
        status,output = subprocess.getstatusoutput(cmd)
        if status!=0:
            raise Exception(output)