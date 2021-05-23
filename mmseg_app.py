#!/usr/bin/env python3
import os
import sys
import time
import pickle
import random
import string
import shutil
import numpy as np
import nibabel as nib
from subprocess import Popen, PIPE
import argparse
import pandas as pd

import tkinter
from tkinter import *
from tkinter import font
from tkinter import filedialog, ttk, PhotoImage
from tkinter import Menu
from tkinter.scrolledtext import ScrolledText
import tkinter.messagebox
import tkinter.font as tkFont
from tkinter.font import Font

import matplotlib
print(matplotlib.matplotlib_fname())
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from nifti_tools import save_nifti
from transp_imshow import transp_imshow
from convertMRCCentreMaskToBinary import convertMRCCentreMaskToBinary
#from utils import scale2D

import mmseg_ll

APP_TITLE='Musclesense workbench'
APP_INSTANCE_ID=''.join(random.choice(string.ascii_letters) for i in range(10))

SETTING_FONT_NAME='Chilanka'
SETTING_FIXED_FONT_NAME="Chilanka"

plt.rcParams["font.family"] = SETTING_FONT_NAME
plt.rcParams['font.size'] = 9

SETTINGS_FILE='mmseg_app.cfg'

IMAGE_TYPE_FAT='Fat'
IMAGE_TYPE_WATER='Water'
IMAGE_TYPE_FATFRACTION='Fat fraction'
IMAGE_TYPE_DIXON345='Dixon 345ms'
IMAGE_TYPE_DIXON460='Dixon 460ms'
IMAGE_TYPE_DIXON575='Dixon 575ms'
IMAGE_TYPE_MASK='Mask'

imagetypes={
    IMAGE_TYPE_FAT:'fatimg',
    IMAGE_TYPE_WATER:'waterimg',
    IMAGE_TYPE_DIXON345:'dixon_345img',
    IMAGE_TYPE_DIXON460:'dixon_460img',
    IMAGE_TYPE_DIXON575:'dixon_575img',
    IMAGE_TYPE_MASK:'maskimg',
}

def displayInfo(info):
    tkinter.messagebox.showinfo('Musclesense workbench information dialog',message=info+' '+'\t'*2)

def displayError(error):
    tkinter.messagebox.showerror('Musclesense workbench error dialog',message=error+' '+'\t'*2)

def askOKCancel(msg):
    return tkinter.messagebox.askokcancel('Musclesense workbench confirmation dialog', msg+' '+'\t'*2, default="cancel", icon="warning")

def askYesNoCancel(msg):
    return tkinter.messagebox.askyesnocancel('Musclesense workbench confirmation dialog', msg+' '+'\t'*2, default="cancel", icon="warning")

def removeAxesImages(ax):
    for child in ax.get_children():
        if type(child) is matplotlib.image.AxesImage:
            if False: print('Removing', child, 'of type', type(child))
            child.remove()

def sliceSelectorGUI(studyToOpen):
    root = tkinter.Tk()
    root.title(APP_TITLE)

    myFont = Font(family=SETTING_FONT_NAME, size=11, weight=tkFont.NORMAL)
    myBoldFont = Font(family=SETTING_FONT_NAME, size=11, weight=tkFont.BOLD)
    myFixedFont = Font(family=SETTING_FIXED_FONT_NAME, size=11, weight=tkFont.NORMAL)

    layout_simple='simple'
    layout_normal='normal'
    layout=layout_normal

    root.fatimg=None
    root.waterimg=None
    root.ffimg=None
    root.dixon_345img=None
    root.dixon_460img=None
    root.dixon_575img=None
    root.maskimg=None
    root.binimg=None
    root.sli=0
    root.selectedSlices=[]
    root.showMask=True
    root.overlayMaskOn=IMAGE_TYPE_FATFRACTION
    root.bin_threshold=50
    root.recentStudies=[]

    def showAbout():
        displayInfo('Musclesense workbench\n\nCopyright 2020, University College London\n\n'
        'Includes icons by Icon8 (https://icons8.com) and code by RemiLehe (https://github.com/RemiLehe/transparent_imshow/blob/master/transparent_imshow/transparent_imshow.py)\n\n'
        'Supported by the Wellcome Trust, and University College London Hospitals NHS Foundation Trust')

    def overlayMaskOn(image_type):
        if root.overlayMaskOn==image_type: return
        root.overlayMaskOn=image_type
        showSlice()

    def prepForDisplay(img,slice,empty_shape=(51,24)):
        if img is None: return np.zeros(empty_shape,dtype=np.uint8).transpose()
        return np.fliplr(np.flipud(img[:,:,slice].transpose()))

    #def prepForDisplayOP(img,slice,empty_shape=(51,24)):
    #    if img is None: return np.fliplr(np.flipud(np.zeros(empty_shape,dtype=np.uint8).transpose()))
    #    return np.fliplr(np.flipud(scale2D(np.array(img)[:,img.shape[1]//2,:],img.shape[0],img.shape[1],order=0,           mode='nearest',prefilter=False).transpose()))

    def showHideMask():
        root.showMask=not root.showMask
        showSlice()

    def flashShowHideMaskButton():
        if root.showMask:
            showHideMaskButton['bg']=firstSliceButton['bg']
            setattr(showHideMaskButton,'isFlashing',0)
            return
            
        if showHideMaskButton['bg']!='orange': showHideMaskButton['bg']='orange'
        else:
            showHideMaskButton['bg']=firstSliceButton['bg']

        root.after(800, flashShowHideMaskButton)

    def showSlice(justmask=False):
        start_time = time.time()

        if root.showMask:
            if showHideMaskButton['text']!='Hide mask':
                showHideMaskButton['text']='Hide mask'
        else:
            if not hasattr(showHideMaskButton,'isFlashing'):
                setattr(showHideMaskButton,'isFlashing',0)
            if not getattr(showHideMaskButton,'isFlashing'):
                showHideMaskButton['text']='Show mask'
                setattr(showHideMaskButton,'isFlashing',1)
                root.after(800, flashShowHideMaskButton)

        if root.fatimg is None:
            text=''
        elif root.sli in root.selectedSlices:
            text=' (slice %d/%d [selected])'%(root.sli+1,root.num_slices)
        else:
            text=' (slice %d/%d)'%(root.sli+1,root.num_slices)

        if not justmask:
            if layout==layout_normal:
                removeAxesImages(ax1)
                root.ai1=ax1.imshow(prepForDisplay(root.fatimg,root.sli),cmap='gray')
                ax1.set_title('Fat'+text)

                removeAxesImages(ax2)
                ax2.imshow(prepForDisplay(root.waterimg,root.sli),cmap='gray')
                ax2.set_title('Water'+text)

        removeAxesImages(ax3)
        if root.overlayMaskOn==IMAGE_TYPE_FATFRACTION: 
            overlayBaseImage=root.ffimg
            add_params={'vmin':0,'vmax':100}
        else:
            overlayBaseImage=getattr(root,imagetypes[root.overlayMaskOn])
            add_params={}

        if overlayBaseImage is None and root.binimg is not None:
            ax3.imshow(prepForDisplay(overlayBaseImage,root.sli,empty_shape=root.binimg.shape[0:2]),cmap='gray')
        else:
            ax3.imshow(prepForDisplay(overlayBaseImage,root.sli),cmap='gray',**add_params)
 
        if root.binimg is not None and root.showMask:
            cmap='winter' if root.sli in root.selectedSlices else 'summer'
            transp_imshow(ax3,prepForDisplay(root.binimg,root.sli),cmap=cmap,alpha=0.3)

        ax3.set_title(root.overlayMaskOn+' & Muscle mask'+text)

        if not justmask:
            removeAxesImages(ax4)
            ax4.imshow(prepForDisplay(root.dixon_345img,root.sli),cmap='gray')
            ax4.set_title('Dixon 345ms'+text)

            removeAxesImages(ax5)
            ax5.imshow(prepForDisplay(root.dixon_460img,root.sli),cmap='gray')
            ax5.set_title('Dixon 460ms'+text)

            removeAxesImages(ax6)
            ax6.imshow(prepForDisplay(root.dixon_575img,root.sli),cmap='gray')
            ax6.set_title('Dixon 575ms'+text)
        
        canvas.draw()
        #toolbar.update()
        print('Update time: {} seconds'.format(round(time.time() - start_time, 2)))

    def gotoSlice(sli):
        if sli>=0 and sli<root.num_slices:
            root.sli=sli
            showSlice()

    def nextSlice():
        root.sli+=1
        if root.sli>=root.num_slices:
            root.sli=root.num_slices-1
        showSlice()

    def prevSlice():
        root.sli-=1
        if root.sli<0:
            root.sli=0
        showSlice()

    def firstSlice():
        root.sli=0
        showSlice()

    def lastSlice():
        root.sli=root.num_slices-1
        showSlice()

    def select():
        if root.sli in root.selectedSlices: root.selectedSlices.remove(root.sli)
        else:
            root.selectedSlices.append(root.sli)
        showSlice()
        calcStatistics()

    def doClose():
       saveSettings()
       root.title('Cleaning up...')
       os.system('rm -rf tmp.'+APP_INSTANCE_ID+'.*')
       root.destroy()

    def addToRecentStudies(res):
        if 'tmp.' not in res and res not in root.recentStudies:
            root.recentStudies.append(res)
            recentStudiesMenu.add_command(label=res, command=lambda:loadStudy(res))

    def saveSettings(verbose=False):
        try:
            settings_data={}
            settings_data['lastdir']=root.lastdir if hasattr(root,'lastdir') else ''
            settings_data['SETTING_PATHTOITKSNAP']=textBox_SETTING_PATHTOITKSNAP.get()
            settings_data['recentStudies']=root.recentStudies
            settings_data['overlayMaskOn']=root.overlayMaskOn

            with open(os.path.join(INSTALL_DIR,SETTINGS_FILE),'wb') as f:
                pickle.dump(settings_data,f)

            if verbose:
                displayInfo('Settings have been saved')
        except Exception as e:
            displayError('ERROR: '+str(e))

    def loadSettings():
        if not os.path.exists(os.path.join(INSTALL_DIR,SETTINGS_FILE)):
            return
        try:
            with open(os.path.join(INSTALL_DIR,SETTINGS_FILE),'rb') as f:
                settings_data=pickle.load(f)

            root.lastdir=settings_data['lastdir']
            textBox_SETTING_PATHTOITKSNAP.delete(0,END)
            textBox_SETTING_PATHTOITKSNAP.insert(0,settings_data['SETTING_PATHTOITKSNAP'])

            if 'recentStudies' in settings_data:
                for study in sorted(settings_data['recentStudies']):
                    addToRecentStudies(study)

            if 'overlayMaskOn' in settings_data:
                root.overlayMaskOn = settings_data['overlayMaskOn']
        except Exception as e:
            displayError('ERROR: '+str(e))

    STUDY_LINK_SEPERATOR=' - '
    def getStudyLink():
        studylink=root.title()
        ch_index=studylink.find(STUDY_LINK_SEPERATOR)
        if ch_index<0: return None
        return studylink[ch_index+len(STUDY_LINK_SEPERATOR):]

    def setStudyLink(studylink):
        if studylink is None:
            root.title(APP_TITLE)
        else:
            root.title(APP_TITLE+STUDY_LINK_SEPERATOR+studylink)

    def makeCopy():
        try:
            saveSettings()
            TEMP_FILE='tmp.'+APP_INSTANCE_ID+'.'+''.join(random.choice(string.ascii_letters) for i in range(10))+'.study'

            study_path=getStudyLink()
            if study_path is not None:
                study_path=os.path.dirname(study_path)
                if os.path.isdir(study_path):
                    TEMP_FILE=os.path.join(study_path,TEMP_FILE)

            if saveStudy(TEMP_FILE):
                Popen([__file__,'-study',TEMP_FILE])
        except Exception as e:
            displayError('ERROR: '+str(e))

    def newWorkbench():
        try:
            saveSettings()
            Popen([__file__])
        except Exception as e:
            displayError('ERROR: '+str(e))

    def saveStudy(filename=None):
        if filename is None:
            if hasattr(root,'lastdir'): initialdir=root.lastdir
            else: initialdir=os.getcwd()

            res=filedialog.asksaveasfilename(initialdir=initialdir, title = "Select study file to save into",
                filetypes = (("Study files","*.study"),("All files","*.*")))
            if type(res) is not str or res=='':
                return False
            print(res)
            root.lastdir=os.path.dirname(res)
            print(root.lastdir)
        else:
            res=filename

        try:
            workspace_data={}
            workspace_data['study']=res
            workspace_data['body_part']=str(bodypart_combobox.get())

            for it in imagetypes:
                workspace_data[imagetypes[it]]=getattr(root,imagetypes[it]+'link')['text']

            workspace_data['case_notes']=case_notes_control.get("1.0", tkinter.END)

            workspace_data['sli']=root.sli
            workspace_data['selected_slices']=root.selectedSlices

            with open(res,'wb') as f:
                pickle.dump(workspace_data,f)

            if filename is None: 
                setStudyLink(res)
                addToRecentStudies(res)
            return True
        except Exception as e:
            displayError('ERROR: '+str(e))
            return False

    def createStudyFromFolder():
        if hasattr(root,'lastdir'): initialdir=root.lastdir
        else: initialdir=os.getcwd()

        res=filedialog.askdirectory(initialdir=initialdir, title = "Select a study directory")
        if type(res) is not str or res=='':
            return

        print(res)
        root.lastdir=res
        print(root.lastdir)
        return

        try:
            with open(res,'rb') as f:
                workspace_data=pickle.load(f)

            newStudy(silent=True)

            root.sli=0
            root.showMask=True

            p_old=p_new=''                    
            study_old=workspace_data['study']
            if res!=study_old:
                p_old=os.path.dirname(study_old)
                p_new=os.path.dirname(res)

            bodypart_combobox.set(workspace_data['body_part'])

            for it in imagetypes:
                if imagetypes[it] in workspace_data and workspace_data[imagetypes[it]]!='Not loaded':
                    chooseImage(it,workspace_data[imagetypes[it]].replace(p_old,p_new),callShowSlice=False)
                else:
                    setattr(root,imagetypes[it],None)
                    setattr(root,imagetypes[it]+'obj',None)
                    getattr(root,imagetypes[it]+'link')['text']='Not loaded'

            case_notes_control.delete(1.0,tkinter.END)
            case_notes_control.insert(tkinter.INSERT,workspace_data['case_notes'])

            root.selectedSlices=[]
            root.selectedSlices=list(workspace_data['selected_slices'])

            if len(root.selectedSlices)>0:
                root.sli=sorted(root.selectedSlices)[0]

            setStudyLink(res)
        except Exception as e:
            displayError('ERROR: '+str(e))

        showSlice()
        calcStatistics()

    def chooseBaselineStudy():
        if hasattr(root,'lastdir'): initialdir=root.lastdir
        else: initialdir=os.getcwd()

        res=filedialog.askopenfilename(initialdir=initialdir, title = "Select the baseline study to load",
            filetypes = (("Study files","*.study"),("All files","*.*")))
        if type(res) is not str or res=='':
            return

        print(res)
        if False:
            root.lastdir=os.path.dirname(res)
            print(root.lastdir)
        else:
            print('chooseBaselineStudy() not updating root.lastdir to avoid confusion')

        try:
            TEMP_FILE='tmp.'+APP_INSTANCE_ID+'.'+''.join(random.choice(string.ascii_letters) for i in range(10))+'.csv'
            p=Popen([__file__,'-study',res,'-csv',TEMP_FILE], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            while p.poll() is None:
                time.sleep(0.5)

            output, err = p.communicate()
            if p.returncode!=0:
                raise Exception('Failed with the following output: '+str(output))

            setattr(root,'baseline_stats_df',pd.read_csv(TEMP_FILE))
            os.remove(TEMP_FILE)

            baselineStudylink['text']=res
            calcStatistics()
        except Exception as e:
            displayError('ERROR: '+str(e))

    def loadStudy(studyToOpen=None):
        if studyToOpen is None:
            if hasattr(root,'lastdir'): initialdir=root.lastdir
            else: initialdir=os.getcwd()

            res=filedialog.askopenfilename(initialdir=initialdir, title = "Select the study to open",
                filetypes = (("Study files","*.study"),("All files","*.*")))
            if type(res) is not str or res=='':
                return
        else:
            res=studyToOpen

        print(res)
        root.lastdir=os.path.dirname(res)
        print(root.lastdir)

        try:
            with open(res,'rb') as f:
                workspace_data=pickle.load(f)

            newStudy(silent=True)

            root.sli=0
            root.showMask=True

            p_old=p_new=''                    
            study_old=workspace_data['study']
            if res!=study_old:
                p_old=os.path.dirname(study_old)
                p_new=os.path.dirname(res)

            bodypart_combobox.set(workspace_data['body_part'])

            for it in imagetypes:
                if imagetypes[it] in workspace_data and workspace_data[imagetypes[it]]!='Not loaded':
                    filename=workspace_data[imagetypes[it]]
                    if not os.path.exists(filename) and p_old!=p_new:
                        filename=workspace_data[imagetypes[it]].replace(p_old,p_new)
                    chooseImage(it,filename,callShowSlice=False)
                else:
                    setattr(root,imagetypes[it],None)
                    setattr(root,imagetypes[it]+'obj',None)
                    getattr(root,imagetypes[it]+'link')['text']='Not loaded'

            case_notes_control.delete(1.0,tkinter.END)
            case_notes_control.insert(tkinter.INSERT,workspace_data['case_notes'])

            root.selectedSlices=[]
            root.selectedSlices=list(workspace_data['selected_slices'])

            if len(root.selectedSlices)>0:
                root.sli=sorted(root.selectedSlices)[0]

            setStudyLink(res)
            addToRecentStudies(res)
        except Exception as e:
            displayError('ERROR: '+str(e))

        showSlice()
        calcStatistics()

    def newStudy(silent=False):
        if not silent:
            choice=askYesNoCancel('Do you wish to save the current study?')
            if choice is None: return
            if choice and not saveStudy(): return

        root.sli=0
        bodypart_combobox.set('')

        root.binimg=None
        root.ffimg=None
        root.showMask=True

        for it in imagetypes:
            setattr(root,imagetypes[it],None)
            getattr(root,imagetypes[it]+'link')['text']='Not loaded'
 
        case_notes_control.delete(1.0,tkinter.END)
 
        root.selectedSlices=[]
        setStudyLink(None)

        if not silent:
            showSlice()
            calcStatistics()

    def calcStatistics():
        if root.ffimg is None or root.binimg is None:
            report='Need fat and water images and a muscle mask to calculate statistics'
            statistics_control.delete(1.0,tkinter.END)
            #statistics_control.insert(tkinter.INSERT,report)
            return

        if root.selectedSlices is None or len(root.selectedSlices)<1:
            report='No slices are selected'
            statistics_control.delete(1.0,tkinter.END)
            statistics_control.insert(tkinter.INSERT,report)
            return
            
        if not np.array_equal(root.fatimgobj.header.get_zooms(),root.maskimgobj.header.get_zooms()):
            report='Fat and mask image resolutions are different'
            statistics_control.delete(1.0,tkinter.END)
            statistics_control.insert(tkinter.INSERT,report)
            return

        slice_resolution=root.fatimgobj.header.get_zooms()[:2]
        slice_resolution=np.prod(slice_resolution)

        if not np.array_equal(np.unique(root.binimg),[0,1]):
            report="Binarised mask unique values not 0 and 1"
            statistics_control.delete(1.0,tkinter.END)
            statistics_control.insert(tkinter.INSERT,report)
            return

        sep='\t'
        if args.csv.strip().lower()!='none': sep=','

        if sep=='\t':
            report='Slice\tMuscle area\t\tFat fraction\n'
        else:
            report='slice,muscle_area,ff_mean,ff_std\n'
        areas=[]
        ffs=[]
        for sli in sorted(root.selectedSlices):
            thisffslice=root.ffimg[:,:,sli]
            thisbinslice=root.binimg[:,:,sli]

            this_area=np.sum(thisbinslice==1)
            if this_area==0: this_area=np.nan
            else:
                 this_area*=slice_resolution
            areas.append(this_area)

            tmp=thisffslice[thisbinslice==1]
            this_ff=np.nanmean(tmp[np.isfinite(tmp)])
            this_ff_std=np.nanstd(tmp[np.isfinite(tmp)])
            ffs.append(this_ff)

            if sep=='\t':
                report+='%d\t%.0f\t\t%.1f±%.1f\n'%(sli+1,this_area,this_ff,this_ff_std)
            else:
                report+='%d,%f,%f,%f\n'%(sli+1,this_area,this_ff,this_ff_std)

        if True:
            graphs_ax1.clear()
            graphs_ax2.clear()
            long_report=''

            if len(root.selectedSlices)>0 and hasattr(root,'baseline_stats_df'):
                df=getattr(root,'baseline_stats_df')
                if len(root.selectedSlices)!=len(df.slice.values):
                    long_report+='CAUTION: different number of slices selected in this time point and the baseline (%d vs %d)\n\n'%(len(root.selectedSlices),len(df.slice.values))
                elif np.sum(np.isfinite(areas))!=np.sum(np.isfinite(df.muscle_area)):
                    long_report+='CAUTION: different number of slices have valid measurements in this time point and the baseline (%d vs %d)\n\n'%(np.sum(np.isfinite(areas)),np.sum(np.isfinite(df.muscle_area)))
                
                this_area_mean=np.nanmean(areas)
                baseline_area_mean=np.nanmean(df.muscle_area)

                this_ff_mean=np.nanmean(ffs)
                baseline_ff_mean=np.nanmean(df.ff_mean)

                area_change=this_area_mean-baseline_area_mean
                ff_change=this_ff_mean-baseline_ff_mean

                long_report+='Baseline mean muscle area\t\t%.0f mm^2\n'%(baseline_area_mean)
                long_report+='Current mean muscle area\t\t%.0f mm^2\n'%(this_area_mean)
                long_report+='Change %.0f mm^2 (%.1f%% of baseline)\n'%(area_change,100*area_change/baseline_area_mean)
                long_report+='\n'

                long_report+='Baseline mean muscle fat fraction\t\t%.1f%%\n'%(baseline_ff_mean)
                long_report+='Current mean muscle fat fraction\t\t%.1f%%\n'%(this_ff_mean)
                long_report+='Change %.1f%% (%.1f%% of baseline)\n'%(ff_change,100*ff_change/baseline_ff_mean)
                long_report+='\n'

                graphs_ax1.bar(['Baseline','This timepoint'],[baseline_area_mean,this_area_mean],color='tab:blue')
                graphs_ax1.set_ylabel('Muscle area (mm^2)')
                graphs_ax2.bar(['Baseline','This timepoint'],[baseline_ff_mean,this_ff_mean],color='tab:red')
                graphs_ax2.set_ylabel('Fat fraction (%)')

            canvas_graphs.draw()

            long_statistics_control.delete(1.0,tkinter.END)
            long_statistics_control.insert(tkinter.INSERT,long_report)

        if sep=='\t':
            report+='AVG\t%.0f±%.0f\t\t%.1f±%.1f\n'%(np.nanmean(areas),np.nanstd(areas),np.nanmean(ffs),np.nanstd(ffs))

        statistics_control.delete(1.0,tkinter.END)
        statistics_control.insert(tkinter.INSERT,report)

    def caseNotesCopyToClipboard():
        root.clipboard_clear()
        root.clipboard_append(case_notes_control.get("1.0", tkinter.END))
        displayInfo("Notes copied to clipboard")

    def statisticsCopyToClipboard():
        root.clipboard_clear()
        root.clipboard_append(statistics_control.get("1.0", tkinter.END))
        displayInfo("Statistics copied to clipboard")

    def editMask():
        if root.fatimg is None or root.waterimg is None or root.dixon_345img is None or root.dixon_460img is None or root.dixon_575img is None or root.maskimg is None:
            displayInfo('Fat, water, Dixon 345ms, Dixon 460ms, Dixon 575ms, and mask images are required')
            return

        fatfile=root.fatimglink['text']
        waterfile=root.waterimglink['text']
        dixon_345file=root.dixon_345imglink['text']
        dixon_460file=root.dixon_460imglink['text']
        dixon_575file=root.dixon_575imglink['text']
        maskfile=root.maskimglink['text']

        if False and 'p3-072-thigh' in getStudyLink():
            temp=root.maskimg.copy()
            maskfile='tmp.'+APP_INSTANCE_ID+'.'+''.join(random.choice(string.ascii_letters) for i in range(10))+'.nii.gz'
            temp=np.roll(temp,[436-426-1,220-212-2],axis=[0,1])
            save_nifti(temp,affine=root.fatimgobj.affine,header=root.fatimgobj.header,filename=maskfile)

        if getMaskType(root.maskimg)==MASK_TYPE_PROBABILISTIC:
            maskfile='tmp.'+APP_INSTANCE_ID+'.'+''.join(random.choice(string.ascii_letters) for i in range(10))+'.nii.gz'
            save_nifti(root.binimg,affine=root.maskimgobj.affine,header=root.maskimgobj.header,filename=maskfile)

        try:
            snap=textBox_SETTING_PATHTOITKSNAP.get()
            Popen([snap,'-g',fatfile,'-s',maskfile,'-o',waterfile,dixon_345file,dixon_460file,dixon_575file])
        except Exception as e:
            displayError('ERROR: '+str(e))

    def generateMask():
        if root.fatimg is None or root.waterimg is None or root.dixon_345img is None or root.dixon_460img is None or root.dixon_575img is None:
            displayInfo('Fat, water, Dixon 345ms, Dixon 460ms, and Dixon 575ms images are required to generate a muscle mask')
            return

        body_part=str(bodypart_combobox.get()).lower()
        if body_part!='calf' and body_part!='thigh':
            displayInfo('Please select the body part from the combination box')
            return

        if not askOKCancel('Please confirm if you have selected the right body part from the combination box'):
            return

        prev_text=generateMaskButton['text']
        prev_bg=generateMaskButton['bg']
        generateMaskButton['bg']='red'
        generateMaskButton['text']='Copying files...'
        generateMaskButton.update()

        try:
            TEMP_DIR='tmp.'+APP_INSTANCE_ID+'.'+''.join(random.choice(string.ascii_letters) for i in range(10))+'.dir'
            os.mkdir(TEMP_DIR)
            nib.save(root.fatimgobj,os.path.join(TEMP_DIR,'fat.nii.gz'))
            nib.save(root.waterimgobj,os.path.join(TEMP_DIR,'water.nii.gz'))
            nib.save(root.dixon_345imgobj,os.path.join(TEMP_DIR,'dixon345.nii.gz'))
            nib.save(root.dixon_460imgobj,os.path.join(TEMP_DIR,'dixon460.nii.gz'))
            nib.save(root.dixon_575imgobj,os.path.join(TEMP_DIR,'dixon575.nii.gz'))

            mmseg_ll.main(body_part,TEMP_DIR,generateMaskButton)

            generateMaskButton['text']='Completing...'
            generateMaskButton.update()
            os.remove(os.path.join(TEMP_DIR,'fat.nii.gz'))
            os.remove(os.path.join(TEMP_DIR,'water.nii.gz'))
            os.remove(os.path.join(TEMP_DIR,'dixon345.nii.gz'))
            os.remove(os.path.join(TEMP_DIR,'dixon460.nii.gz'))
            os.remove(os.path.join(TEMP_DIR,'dixon575.nii.gz'))

            dirToSaveTo=getStudyLink()
            if dirToSaveTo is None: dirToSaveTo=os.getcwd()
            else:
                dirToSaveTo=os.path.dirname(dirToSaveTo)
    
            saveNum=0
            while True:
                saveNum+=1
                fname=os.path.join(dirToSaveTo,'cnn-'+body_part+'-take-'+str(saveNum)+'.nii.gz')
                if not os.path.exists(fname):
                    break

            target_file=os.path.join(dirToSaveTo,'cnn-'+body_part+'-take-'+str(saveNum)+'.nii.gz')
            shutil.move(os.path.join(TEMP_DIR,'cnn-'+body_part+'.nii.gz'),
                target_file)

            displayInfo('The generated mask has been saved as '+target_file)

            shutil.move(os.path.join(TEMP_DIR,'std-'+body_part+'.nii.gz'),
                os.path.join(dirToSaveTo,'std-'+body_part+'-take-'+str(saveNum)+'.nii.gz'))

            os.rmdir(TEMP_DIR)
        except Exception as e:
            displayError('Could not generate the muscle mask\n\n'+'ERROR: '+str(e))

        generateMaskButton['text']=prev_text
        generateMaskButton['bg']=prev_bg

    MASK_TYPE_BINARY=0
    MASK_TYPE_PROBABILISTIC=1
    MASK_TYPE_MULTICLASS=2

    def getMaskType(maskimg):
        if set(np.unique(maskimg)).issubset([0,1]):
            return MASK_TYPE_BINARY
        if np.array_equal(maskimg,maskimg.astype(np.uint16)):
            return MASK_TYPE_MULTICLASS
        return MASK_TYPE_PROBABILISTIC

    def convertMaskToBinary(maskimg):
        binimg=maskimg.copy()
        mask_type=getMaskType(binimg)
        if mask_type==MASK_TYPE_BINARY: 
            print('Assuming a binary mark')
        elif mask_type==MASK_TYPE_PROBABILISTIC:
            print('Assuming a probabilistic mask')
            binimg[binimg>root.bin_threshold/100.0]=1
            binimg[binimg<=root.bin_threshold/100.0]=0
        elif mask_type==MASK_TYPE_MULTICLASS:
            print('Assuming a multiclass mask')
            body_part=str(bodypart_combobox.get()).strip().lower()
            if len(body_part)==0: 
                raise Exception('Cannot convert this mask to binary before the right body part is selected from the combination box')
            maskfilename=root.maskimglink['text']
            if not convertMRCCentreMaskToBinary(maskfilename,body_part,binimg):
                raise Exception('Could not convert mask to binary')
        else:
            raise Exception('Unknown mask type '+str(mask_type))
        return binimg

    def statistics_controlOnClick(event):
        line_text = statistics_control.get("current linestart","current lineend")
        str_tokens = line_text.split('\t')
        try:
            if len(str_tokens)>0:
                gotoSlice(int(str_tokens[0])-1)
        except:
            pass

    def bin_threshold_sliderOnClick_GO():
        try:
            root.binimg=convertMaskToBinary(root.maskimg)
            showSlice(justmask=True)
            calcStatistics()
        except Exception as ex:
            displayError(str(ex))
        setattr(bin_threshold_slider,'updateQueued',0)

    def bin_threshold_sliderOnClick(event):
        if root.bin_threshold==bin_threshold_slider.get(): return

        root.bin_threshold=bin_threshold_slider.get()
        bin_threshold_label['text']='Binarisation threshold (%d%%): '%(root.bin_threshold)

        if root.maskimg is None: return

        if hasattr(bin_threshold_slider,'updateQueued'):
            if getattr(bin_threshold_slider,'updateQueued')==1: return

        setattr(bin_threshold_slider,'updateQueued',1)
        root.after(100, bin_threshold_sliderOnClick_GO)
        
    def chooseImage(it,filename,callShowSlice=True):
        if filename is None:
            if hasattr(root,'lastdir'): initialdir=root.lastdir
            else: initialdir=os.getcwd()

            res=filedialog.askopenfilename(initialdir=initialdir, title = "Select "+it+" image",
                filetypes = (("NIFTI files","*.nii*"),("All files","*.*")))
            if type(res) is not str or res=='':
                return
            print(res)
            root.lastdir=os.path.dirname(res)
            print(root.lastdir)
        else:
            res=filename
        getattr(root,imagetypes[it]+'link')['text']=res
        try:
            setattr(root,imagetypes[it],None)
            setattr(root,imagetypes[it]+'obj',None)
            nibobj=nib.load(res)
            nibdata=nibobj.get_data()

            if len(nibdata.shape)!=3: 
                raise Exception('Shape is not 3D, it is '+str(nibdata.shape))
            
            for oit in imagetypes:
                if oit==it: continue
                other_nibdata=getattr(root,imagetypes[oit])
                if other_nibdata is None: continue
                if nibdata.shape!=other_nibdata.shape:
                    raise Exception("%s and %s shapes do not match: %s vs %s"%(it,oit,str(nibdata.shape),str(other_nibdata.shape)))

            setattr(root,imagetypes[it]+'obj',nibobj)
            setattr(root,imagetypes[it],nibdata)
            root.num_slices=getattr(root,imagetypes[it]).shape[2]

            if bodypart_combobox.get()=='':
                res_lower=res.lower()
                if '/cf/' in res_lower or 'calf' in res_lower:
                    bodypart_combobox.set('Calf')
                elif '/th/' in res_lower or 'thigh' in res_lower:
                    bodypart_combobox.set('Thigh')
        except Exception as e:
            getattr(root,imagetypes[it]+'link')['text']='ERROR: '+str(e).replace('\n','')

        if it=='Fat' or it=='Water':
            root.ffimg=None
            try:
                if root.fatimg is not None and root.waterimg is not None:
                    root.ffimg=root.fatimg/(root.fatimg+root.waterimg)*100
            except Exception as e:
                displayError('ERROR: '+str(e))
        elif it=='Mask':
            root.binimg=None
            try:
                if root.maskimg is not None:
                    root.binimg=convertMaskToBinary(root.maskimg)
                    root.selectedSlices=list(range(0,root.maskimg.shape[2]))
                    segmentedSlices=list(np.max(root.binimg,axis=(0,1))>0)
                    if True in segmentedSlices:
                        root.sli=segmentedSlices.index(True)
            except Exception as e:
                displayError('ERROR: '+str(e))
        elif 'dixon' in it.lower():
            if getattr(root,imagetypes[it]) is not None:
                if not mmseg_ll.checkDixonImage(getattr(root,imagetypes[it])):
                    displayInfo("WARNING: The loaded image did not pass verification checks and may be a phase image")
            
        if callShowSlice: 
            showSlice()
            calcStatistics()

    fig = Figure(facecolor=root['bg'])
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.98, top=0.93, wspace=0.05, hspace=0.25)

    if layout==layout_simple:
        ax3=fig.add_subplot(2,1,1)   
        ax4=fig.add_subplot(2,1,2)   
    else:
        ax1=fig.add_subplot(2,3,1)   
        ax2=fig.add_subplot(2,3,2)   
        ax3=fig.add_subplot(2,3,3)   
        ax4=fig.add_subplot(2,3,4)   
        ax5=fig.add_subplot(2,3,5)   
        ax6=fig.add_subplot(2,3,6)   
        for ax in (ax1,ax2,ax3,ax4,ax5,ax6):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig_graphs = Figure(facecolor=root['bg'])
    fig_graphs.subplots_adjust(left=0.10, bottom=0.07, right=0.94, top=0.99, wspace=0.32, hspace=0.25)
    graphs_ax1=fig_graphs.add_subplot(1,2,1)
    graphs_ax2=fig_graphs.add_subplot(1,2,2)

    tab_parent=ttk.Notebook(root)
    
    tab_home=tkinter.Frame(tab_parent)
    tab_graphs=tkinter.Frame(tab_parent)
    tab_notes=tkinter.Frame(tab_parent)
    tab_settings=tkinter.Frame(tab_parent)
    
    tab_parent.add(tab_home,text='Home')
    tab_parent.add(tab_notes,text='Study notes')
    tab_parent.add(tab_graphs,text='Analysis')
    tab_parent.add(tab_settings,text='Settings')
    tab_parent.pack(expand=1,fill='both')

    canvas = FigureCanvasTkAgg(fig, master=tab_home)
    #toolbar = NavigationToolbar2Tk(canvas, tab_home)

    canvas_graphs = FigureCanvasTkAgg(fig_graphs,master=tab_graphs)

    frame2=tkinter.Frame(tab_home)
    frame2.pack(side=tkinter.BOTTOM)
    frame1=tkinter.Frame(tab_home)
    frame1.pack(side=tkinter.BOTTOM)
    
    firstSliceButton = tkinter.Button(master=tab_home, text="First slice", command=firstSlice)
    firstSliceButton.pack(in_=frame1,side=tkinter.LEFT)
    firstSliceButton.configure(font=myFont)
    button = tkinter.Button(master=tab_home, text="Last slice", command=lastSlice)
    button.pack(in_=frame1,side=tkinter.LEFT)
    button.configure(font=myFont)
    button = tkinter.Button(master=tab_home, text="Previous slice", command=prevSlice)
    button.pack(in_=frame1,side=tkinter.LEFT)
    button.configure(font=myFont)
    button = tkinter.Button(master=tab_home, text="Next slice", command=nextSlice)
    button.pack(in_=frame1,side=tkinter.LEFT)
    button.configure(font=myFont)
    selectButton = tkinter.Button(master=tab_home, text="Select/Deselect slice", command=select)
    selectButton.pack(in_=frame1,side=tkinter.LEFT)
    selectButton.configure(font=myFont)

    generateMaskButton = tkinter.Button(master=tab_home, text="Generate mask", command=generateMask)
    generateMaskButton.pack(in_=frame2,side=tkinter.LEFT)
    generateMaskButton.configure(font=myFont)
    button = tkinter.Button(master=tab_home, text="Edit mask", command=editMask)
    button.pack(in_=frame2,side=tkinter.LEFT)
    button.configure(font=myFont)
    showHideMaskButton = tkinter.Button(master=tab_home, text="Hide mask", command=showHideMask)
    showHideMaskButton.pack(in_=frame2,side=tkinter.LEFT)
    showHideMaskButton.configure(font=myFont)

    root.option_add('*TCombobox*Listbox.font', myFont)
    root.option_add('*Dialog.msg.font', myFont)

    icon_select=PhotoImage(file=os.path.join(INSTALL_DIR,'icons8-open-door-16.png'))
    icon_delete=PhotoImage(file=os.path.join(INSTALL_DIR,'icons8-erase-24.png'))
    icon_copy=PhotoImage(file=os.path.join(INSTALL_DIR,'icons8-copy-24.png'))

    frame=tkinter.Frame(tab_home)
    frame.pack(side=tkinter.TOP,anchor="w")
    frame.pack(padx=20,pady=20)

    row=0
    label=tkinter.Label(frame,text=' Anatomy: ')
    label.grid(row=row,column=1,sticky='w')
    label.configure(font=myFont)
    bodypart_combobox=ttk.Combobox(frame,state='readonly',values=['Thigh','Calf'])
    bodypart_combobox.grid(row=row,column=2,sticky='w')
    bodypart_combobox.configure(font=myFont)
    row+=1
    
    for it in imagetypes:
        button=tkinter.Button(frame,image=icon_select,command=lambda it=it: chooseImage(it,None))
        button.grid(row=row,sticky='w')
        label=tkinter.Label(frame)
        label.grid(row=row,column=1,sticky='w')
        label['text']=' '+it+' image: '
        label.configure(font=myFont)
        link = tkinter.Label(master=frame, text="Not loaded")
        link.configure(font=myFont)
        link.grid(row=row,column=2,sticky='w')
        setattr(root,imagetypes[it]+'link',link)
        row+=1

    label=tkinter.Label(frame,text='   Statistics:')
    label.grid(row=0,column=3,sticky='e')
    label.configure(font=myBoldFont)
    label=tkinter.Label(frame,text=' ')
    label.grid(row=0,column=4,sticky='e')
    button=tkinter.Button(frame,image=icon_copy,command=statisticsCopyToClipboard)
    button.grid(row=1,column=3,sticky='e',rowspan=2)

    statistics_control=ScrolledText(frame,width=60,height=10)
    statistics_control.grid(row=0,column=5,rowspan=7)
    statistics_control.configure(font=myFixedFont)
    statistics_control.bind('<Button-1>',statistics_controlOnClick)

    label=tkinter.Label(tab_notes,text=' ')
    label.grid(row=0,column=0,sticky='e')
    label=tkinter.Label(tab_notes,text='   Notes:')
    label.grid(row=1,column=0,sticky='e')
    label.configure(font=myBoldFont)
    button=tkinter.Button(tab_notes,image=icon_delete,command=lambda:case_notes_control.delete(1.0,tkinter.END))
    button.grid(row=2,column=0,sticky='e')
    button=tkinter.Button(tab_notes,image=icon_copy,command=caseNotesCopyToClipboard)
    button.grid(row=3,column=0,sticky='e')
    label=tkinter.Label(tab_notes,text=' ')
    label.grid(row=1,column=1,sticky='e')

    case_notes_control=ScrolledText(tab_notes,width=80,height=20)
    case_notes_control.grid(row=1,column=2,rowspan=80)
    case_notes_control.configure(font=myFont)

    label=tkinter.Label(tab_settings,text=' ')
    label.grid(row=0,column=0,sticky='e')
    label=tkinter.Label(tab_settings,text='Path to ITK-SNAP: ')
    label.grid(row=1,column=1,sticky='w')
    label.configure(font=myFont)
    textBox_SETTING_PATHTOITKSNAP=tkinter.Entry(tab_settings,width=80)
    textBox_SETTING_PATHTOITKSNAP.grid(row=1,column=2,sticky='w')
    textBox_SETTING_PATHTOITKSNAP.configure(font=myFont)
    textBox_SETTING_PATHTOITKSNAP.insert(0,'itksnap') # /home/bkanber/itksnap-3.6.0-20170401-Linux-x86_64/bin/itksnap

    bin_threshold_label=tkinter.Label(tab_settings,text='Binarisation threshold (%d%%): '%(root.bin_threshold))
    bin_threshold_label.grid(row=2,column=1,sticky='w')
    bin_threshold_label.configure(font=myFont)
    bin_threshold_slider=tkinter.Scale(tab_settings,from_=10,to=90,bigincrement=10,resolution=10,showvalue=0,orient=tkinter.HORIZONTAL)
    bin_threshold_slider.set(root.bin_threshold)
    bin_threshold_slider.grid(row=2,column=2,sticky='w')
    bin_threshold_slider.configure(font=myFont)
    bin_threshold_slider.bind('<ButtonRelease-1>',bin_threshold_sliderOnClick)
    bin_threshold_slider.bind('<B1-Motion>',bin_threshold_sliderOnClick)
    bin_threshold_label.grid_forget()
    bin_threshold_slider.grid_forget()

    label=tkinter.Label(tab_settings,text=' ')
    label.grid(row=2,column=0,sticky='e')
    button=tkinter.Button(tab_settings,text='Save settings',command=lambda:saveSettings(verbose=True))
    button.grid(row=3,column=2,sticky='e')
    button.configure(font=myFont)

    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    frame=tkinter.Frame(tab_graphs)
    frame.pack(side=tkinter.TOP,anchor="w")
    frame.pack(padx=20,pady=(20,0))
    button=tkinter.Button(frame,image=icon_select,command=chooseBaselineStudy)
    button.grid(row=0,column=0,sticky='w')
    label=tkinter.Label(frame,text=' Baseline study: ')
    label.grid(row=0,column=1,sticky='w')
    label.configure(font=myBoldFont)
    baselineStudylink = tkinter.Label(master=frame, text="Not loaded")
    baselineStudylink.configure(font=myFont)
    baselineStudylink.grid(row=0,column=2,sticky='w')
    label=tkinter.Label(frame,text=' ')
    label.grid(row=1,column=1,sticky='w')
    label.configure(font=myFont)

    longgraphs_frame=tkinter.Frame(tab_graphs)
    longgraphs_frame.pack(side=tkinter.BOTTOM,anchor="w",fill=tkinter.BOTH, expand=1)
    longgraphs_frame.pack(padx=20,pady=20)
    long_statistics_control=ScrolledText(longgraphs_frame,width=120,height=18)
    long_statistics_control.pack(side=tkinter.LEFT,anchor='n')
    long_statistics_control.configure(font=myFixedFont)
    long_statistics_control.bind('<Button-1>',statistics_controlOnClick)
    button=tkinter.Button(longgraphs_frame,image=icon_copy,command=statisticsCopyToClipboard)
    button.pack(side=tkinter.LEFT,anchor='n')
    label=tkinter.Label(longgraphs_frame,text='    ')
    label.pack(side=tkinter.LEFT)

    canvas_graphs.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)

    menubar = Menu(root)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="New study", command=newStudy)
    filemenu.add_command(label="Load study", command=loadStudy)
    recentStudiesMenu = Menu(filemenu, tearoff=0)
    filemenu.add_cascade(label="Load recent...", menu=recentStudiesMenu)
    filemenu.add_command(label="Save study", command=saveStudy)
    filemenu.add_command(label="Make a copy", command=makeCopy)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=doClose)
    menubar.add_cascade(label="File", menu=filemenu)

    overlayMaskOnMenu = Menu(menubar, tearoff=0)
    overlayMaskOnMenu.add_command(label="Fat image", command=lambda:overlayMaskOn(IMAGE_TYPE_FAT))
    overlayMaskOnMenu.add_command(label="Water image", command=lambda:overlayMaskOn(IMAGE_TYPE_WATER))
    overlayMaskOnMenu.add_command(label="Fat fraction image", command=lambda:overlayMaskOn(IMAGE_TYPE_FATFRACTION))
    overlayMaskOnMenu.add_command(label="Dixon 345ms image", command=lambda:overlayMaskOn(IMAGE_TYPE_DIXON345))
    overlayMaskOnMenu.add_command(label="Dixon 460ms image", command=lambda:overlayMaskOn(IMAGE_TYPE_DIXON460))
    overlayMaskOnMenu.add_command(label="Dixon 575ms image", command=lambda:overlayMaskOn(IMAGE_TYPE_DIXON575))

    viewmenu = Menu(menubar, tearoff=0)
    viewmenu.add_command(label="New workbench", command=newWorkbench)
    viewmenu.add_cascade(label="Overlay mask on...", menu=overlayMaskOnMenu)
    menubar.add_cascade(label="View", menu=viewmenu)

    #toolsmenu = Menu(menubar, tearoff=0)
    #toolsmenu.add_command(label="Create study from folder", command=createStudyFromFolder)
    #menubar.add_cascade(label="Tools", menu=toolsmenu)

    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label="About", command=showAbout)
    menubar.add_cascade(label="Help", menu=helpmenu)

    root.config(menu=menubar)
    root.protocol("WM_DELETE_WINDOW", doClose)
    root.iconphoto(False, PhotoImage(file=os.path.join(INSTALL_DIR,'M-icon-32.png')))

    loadSettings()

    if studyToOpen.lower().strip()=='new':
        showSlice()
    else:
        loadStudy(studyToOpen)
        if args.csv.strip().lower()!='none':
            try:
                with open(args.csv,'wt') as f:
                    f.write(statistics_control.get("1.0", tkinter.END))
                print('CSV file %s written'%(args.csv))
                return 0
            except Exception as ex:
                print(str(ex))
                return 1

    tkinter.mainloop()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser("mmseg_app")
    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('-study', type=str, help='Study to open (default: new)', default='new')
    parser.add_argument('-csv', type=str, help='CSV file to export stats to (default: none)', default='none')
    args=parser.parse_args()

    MODULE_NAME=os.path.basename(__file__)
    INSTALL_DIR=os.path.dirname(os.path.realpath(__file__))
    print('INSTALL_DIR',INSTALL_DIR,MODULE_NAME)

    sys.exit(sliceSelectorGUI(args.study))
