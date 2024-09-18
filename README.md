## Musclesense 
Musclesense is a trained neural network for the anatomical segmentation of muscles in 3-point Dixon, T1w, and T2-stir, lower-limb MRI volumes. 

## Example
![image](https://github.com/user-attachments/assets/647b7253-4b80-44e6-b8b3-81a7829a2b04)

## Installation
The instructions below are for the Linux OS. Please replace ```<INSTALL_DIR>``` with the path to the folder you wish to install the software.

* Download the latest [release version](https://github.com/bariskanber/musclesenseworkbench/releases) of Musclesense.

* Extract the zip file in ```<INSTALL_DIR>```.

* Install [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install) for Linux in ```<INSTALL_DIR>/miniconda3```. Please note that Musclesense requires Python version > 3.12.

* Install the required Python modules by running ```<INSTALL_DIR>/miniconda3/bin/python -m pip install -r <INSTALL_DIR>/requirements.txt```

## Getting started

### Example 1
Run the following command to perform individual muscle segmentation on the two T1w calf datasets in the directory <test_dir>.

```
<INSTALL_DIR>/miniconda3/bin/python mmseg_ll.py -al calf -modalities t1 -inputdir <test_dir>
```

<test_dir> is expected to have the following directory structure:

```
<test_dir>/
├── subject1/
│   └── t1.nii.gz
├── subject2/
│   └── t1.nii.gz
```

A file labelled calf_parcellation_t1.nii.gz will be produced in each subject directory.

### Example 2
Run the following command to perform whole muscle segmentation on the two T2-stir thigh datasets in the directory <test_dir2>.

```
<INSTALL_DIR>/miniconda3/bin/python mmseg_ll.py -al thigh -modalities t2_stir -inputdir <test_dir2> --wholemuscle
```

<test_dir2> is expected to have the following directory structure:

```
<test_dir2>/
├── subject1/
│   └── t2_stir.nii.gz
├── subject2/
│   └── t2_stir.nii.gz
```

A file labelled thigh_segmentation_t2_stir.nii.gz will be produced in each subject directory.

### Example 3
Run the following command to perform individual muscle segmentation on the two 3-point Dixon calf datasets in the directory <test_dir3>.

```
<INSTALL_DIR>/miniconda3/bin/python mmseg_ll.py -al calf -modalities dixon_345_460_575 -inputdir <test_dir3>
```

<test_dir3> is expected to have the following directory structure:

```
<test_dir3>/
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
