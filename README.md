### Musclesense 
Musclesense is a trained ANN for the segmentation of muscle MR images. 
The following are currently supported: MRI sequences (3-point Dixon, T1w, T2-stir), platforms (Ubuntu), anatomical sites (calf, thigh).

### Example
![image](https://github.com/user-attachments/assets/c3f6438c-c65e-4683-ac8d-024c4d83609f)

### Getting started
Download the latest [release version](https://github.com/bariskanber/musclesenseworkbench/releases).

As an example, run the following command to perform individual muscle segmentation on the two T1w calf datasets in the directory <test_dir>.

```
<INSTALL_DIR>/miniconda3/bin/python mmseg_ll.py -al calf -inputdir <test_dir> --multiclass
```

<test_dir> is expected to have the following directory structure:

```
<test_dir>/
├── subject1/
│   └── t1.nii.gz
├── subject2/
│   └── t1.nii.gz
```

### Funding
We are grateful to the following for their funding and support of this project: Wellcome Trust, National Institute for Health and Care Research, National Brain Appeal.

### Enquiries
Please submit any enquiries [here](mailto:b.kanber@ucl.ac.uk).
