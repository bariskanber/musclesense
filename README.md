### Musclesense 
Musclesense is a trained ANN for the segmentation of muscle MR images. 

The following are currently supported: MRI acquisitions (3-point Dixon, T1w, T2-stir), OSes (Linux), anatomical locations (calf, thigh).

### Example
![image](https://github.com/user-attachments/assets/c3f6438c-c65e-4683-ac8d-024c4d83609f)

### Getting started
Download the latest [release version](https://github.com/bariskanber/musclesenseworkbench/releases). Alternatively, you can download or clone the repository if you would like to test the latest build. 

To run the the software, type ***<INSTALL_DIR>/miniconda3/bin/python mmseg_ll.py -h*** where ***<INSTALL_DIR>*** is the directory where you have downloaded the software.

For example, ***<INSTALL_DIR>/miniconda3/bin/python mmseg_ll.py -al calf -inputdir <DATA_DIR> --multiclass*** performs individual muscle segmentation for the two T1w datasets in the directory <DATA_DIR>. The latter is expected to have the following directory structure:

```
<DATA_DIR>/
├── subject1/
│   └── t1.nii.gz
├── subject2/
│   └── t1.nii.gz
```

### Funding
We are grateful to the following for their funding and support of this project: Wellcome Trust, National Institute for Health and Care Research, National Brain Appeal.

### Enquiries
Please submit any enquiries [here](mailto:b.kanber@ucl.ac.uk)
