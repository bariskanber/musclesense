## Musclesense 
Musclesense is a trained neural network for the anatomical segmentation of muscles in 3-point Dixon, T1w, and T2-stir, lower-limb MRI volumes. 

## Example
![image](https://github.com/user-attachments/assets/647b7253-4b80-44e6-b8b3-81a7829a2b04)

## Installation
The instructions below are for the Linux and install the software at ```~/musclesense```. Replace ```~/musclesense``` with another path if you wish to install the software elsewhere on your filesystem.

* Download the latest [release version](https://github.com/bariskanber/musclesenseworkbench/releases) of Musclesense and extract the zip file in ```~/musclesense```.

* Install [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install) in ```~/musclesense/miniconda3```.

* Install the required Python modules by running: ```~/musclesense/miniconda3/bin/python -m pip install -r ~/musclesense/requirements.txt```

## Getting started

### Example 1
Run the following command to perform individual muscle segmentation on the two T1w calf datasets in the directory /mydir.

```
~/musclesense/miniconda3/bin/python ~/musclesense/mmseg_ll.py -al calf -modalities t1 -inputdir /mydir
```

mydir is expected to have the following directory structure:

```
mydir/
├── subject1/
│   └── t1.nii.gz
├── subject2/
│   └── t1.nii.gz
```

A file labelled calf_parcellation_t1.nii.gz will be produced in each subject directory.

### Example 2
Run the following command to perform whole muscle segmentation on the two T2-stir thigh datasets in the directory /mydir.

```
~/musclesense/miniconda3/bin/python ~/musclesense/mmseg_ll.py -al thigh -modalities t2_stir -inputdir /mydir --wholemuscle
```

mydir is expected to have the following directory structure:

```
mydir/
├── subject1/
│   └── t2_stir.nii.gz
├── subject2/
│   └── t2_stir.nii.gz
```

A file labelled thigh_segmentation_t2_stir.nii.gz will be produced in each subject directory.

### Example 3
Run the following command to perform individual muscle segmentation on the two 3-point Dixon calf datasets in the directory /mydir.

```
~/musclesense/miniconda3/bin/python ~/musclesense/mmseg_ll.py -al calf -modalities dixon_345_460_575 -inputdir /mydir
```

mydir is expected to have the following directory structure:

```
mydir/
├── subject1/
│   └── Dixon345.nii.gz
│   └── Dixon460.nii.gz
│   └── Dixon575.nii.gz
├── subject2/
│   └── Dixon345.nii.gz
│   └── Dixon460.nii.gz
│   └── Dixon575.nii.gz
```

A file labelled calf_parcellation_dixon_345_460_575.nii.gz will be produced in each subject directory.

#### Funding
We are grateful to the Wellcome Trust, National Institute for Health and Care Research, and National Brain Appeal for their kind funding and support of this project.

Please consider citing the following publications if you use this software in your research:
* Musclesense: a trained, artificial neural network for the anatomical segmentation of lower limb magnetic resonance images in neuromuscular diseases (Kanber et al., 2021).
* Quantitative MRI outcome measures in CMT1A using automated lower limb muscle segmentation (O'Donnell et al., 2024).

#### Enquiries
Please submit any enquiries [here](mailto:b.kanber@ucl.ac.uk).
