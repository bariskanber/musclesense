### Musclesense workbench

Musclesense workbench is an open-source software platform for neuromuscular imaging research, that is intended to be community maintained and developed. 

It aims to offer MRI visualisation, processing, cross-sectional/longitudinal analysis, and biomarker extraction capabilities all within one platform (utilising the many great neuromuscular image processing toolboxes that are available out there through an installable plug-in mechanism). 

Please [email](mailto:b.kanber@ucl.ac.uk) us to get involved.

### Background

The software was produced as part of the Wellcome Institutional Strategic Support Fund (ISSF3) – AI in Healthcare Call 2019 project ***“Towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods”***. 

Funded by the Wellcome Trust and the National Institute for Health Research Biomedical Research Centre at University College London and University College London Hospitals NHS Foundation Trust, the project aimed to contribute towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods.

### What it looks like

![Screenshot from 2021-11-11 01-44-54](https://user-images.githubusercontent.com/12815964/141222112-3450396a-e3cd-4f39-a33c-2115ab3c74cf.png)

### Getting started

Download a [release version](https://github.com/bariskanber/musclesenseworkbench/releases) and save at a location of your choice on your Linux workstation. The source code and the model weights should be saved in the same directory. Please do not just clone or download the repository as you will be missing the required model weights.

The following are prerequisites (where appropriate with example commands):   
  
*[itk-snap](http://www.itksnap.org)  
Python3  
python3 -m pip install numpy  
python3 -m pip install nibabel  
sudo apt install python3-tk  
python3 -m pip install matplotlib  
python3 -m pip install pandas  
python3 -m pip install joblib  
python3 -m pip install scikit-learn  
python3 -m pip install tensorflow==2.3.0  
python3 -m pip install keras  
python3 -m pip install scikit-image*  

To run the workbench, type ***python3 mmseg_app.py*** from the installation directory

### Troubleshooting
1.  Symptom: ***KeyError: 'image_dim_ordering'*** Solution: add ***"image_dim_ordering": "th"*** to ***~/.keras/keras.json***

### Notices
Musclesense the algorithm, and Musclesense Workbench should not be used in the diagnosis or treatment of patients.

Includes icons by [Icon8](https://icons8.com)

### Enquiries
Please submit any enquiries [here](mailto:b.kanber@ucl.ac.uk)
