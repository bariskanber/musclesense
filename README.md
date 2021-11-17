### Musclesense workbench

Musclesense workbench is an open-source software for neuromuscular imaging research, that is maintained and developed by a consortium of higher education institutions. 

The software offers cross-platform MRI visualisation, processing, cross-sectional/longitudinal analysis, and biomarker extraction all within one platform. 

### Background

The software was produced as part of the Wellcome Institutional Strategic Support Fund (ISSF3) – AI in Healthcare Call 2019 project ***“Towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods”***. 

Funded by the Wellcome Trust and the National Institute for Health Research Biomedical Research Centre at University College London and University College London Hospitals NHS Foundation Trust, the project aimed to contribute towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods.

Future versions of the software will allow the use of different muscle segmentation tools utilising different methods and/or acquisition sequences using a configurable plug-in mechanism. The default deep-learning-based segmentation tool embedded into the software was described in the article [linked here](https://pubmed.ncbi.nlm.nih.gov/32892313/).

### Consortium
The current consortium members are University College London (UK), and Newcastle University (UK). Please [email](mailto:b.kanber@ucl.ac.uk) us to express interest in joining the consortium.

### What it looks like

![Screenshot from 2021-11-14 02-24-45](https://user-images.githubusercontent.com/12815964/141664991-b521a9a8-9287-4387-b9df-0d4917fe024a.png)

### Getting started

Download the latest [release version](https://github.com/bariskanber/musclesenseworkbench/releases). Create an installation directory/folder (e.g. ***musclesenseworkbench***) at a location of your choice on your computer. The source code (e.g. ***musclesenseworkbench-rX.Y.zip***) and the model weights (e.g. ***musclesenseweights-vA.B.zip***) should then both be unzipped into the installation directory. 

Please do not just clone or download the repository as you will be missing the required model weights.

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
python3 -m pip install scikit-image --user  
python3 setup.py (from the installation directory)*  

To run the workbench, type ***python3 mmseg_app.py*** from the installation directory

While the above example commands are given for installation on an Ubuntu OS, the software should work on other Linux OSes as well as MacOS. It will probably need some modifications before it will work on Windows though. Do not hesitate to get in touch if you have any trouble installing and getting the software up-and-running on your system.

### Notices
Musclesense the algorithm, and Musclesense Workbench should not be used in the diagnosis or treatment of patients.

### Enquiries
Please submit any enquiries [here](mailto:b.kanber@ucl.ac.uk)
