### Musclesense workbench

Musclesense workbench is an open-source software platform for neuromuscular imaging research, that is intended to be community maintained and developed. 

It aims to offer MRI visualisation, processing, cross-sectional/longitudinal analysis, and biomarker extraction capabilities all within one platform (utilising the many great neuromuscular image processing toolboxes that are available out there through an installable plug-in mechanism). 

Please [email](mailto:b.kanber@ucl.ac.uk) us to get involved.

### Background

The software was produced as part of the Wellcome Institutional Strategic Support Fund (ISSF3) – AI in Healthcare Call 2019 project ***“Towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods”***. 

Funded by the Wellcome Trust and the National Institute for Health Research Biomedical Research Centre at University College London and University College London Hospitals NHS Foundation Trust, the project aimed to contribute towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods.

### What it looks like

![Screenshot from 2021-11-11 01-47-16](https://user-images.githubusercontent.com/12815964/141222325-39cfd6cd-06b2-4a21-b3a9-e2cf1856cbd7.png)

### Getting started

Download the latest [release version](https://github.com/bariskanber/musclesenseworkbench/releases) and save at a location of your choice on your Linux workstation. The source code and the model weights should be saved in the same directory. Please do not just clone or download the repository as you will be missing the required model weights.

The following are prerequisites (where appropriate with example commands):   

*[itk-snap](http://www.itksnap.org)  
[Python3](https://www.python.org/downloads/)  
sudo apt install python3-pip  
sudo apt install python3-tk  
python3 -m pip install numpy --user  
python3 -m pip install nibabel --user  
python3 -m pip install matplotlib --user  
python3 -m pip install pandas --user  
python3 -m pip install joblib --user  
python3 -m pip install scikit-learn --user  
python3 -m pip install tensorflow==2.3.0 --user  
python3 -m pip install keras==2.3.1 --user  
python3 -m pip install h5py==2.10.0 --user  
python3 -m pip install scikit-image* --user  

To run the workbench, type ***python3 mmseg_app.py*** from the installation directory

While the above example commands are given for an Ubuntu v20 OS installation, the software should work on other Linux OSes as well as MacOS. It will probably need some modifications before it will work on Windows though. Do not hesitate to get in touch if you have any trouble installing and getting the software up-and-running on your system.

### Notices
Musclesense the algorithm, and Musclesense Workbench should not be used in the diagnosis or treatment of patients.

Includes icons by [Icon8](https://icons8.com)

### Enquiries
Please submit any enquiries [here](mailto:b.kanber@ucl.ac.uk)
