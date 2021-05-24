*Musclesense Workbench* is a software application that supports the processing and analysis of 3-point Dixon MRI of the lower limbs. 

The software was produced as part of the project ***“Towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods”***. 

Funded by the Wellcome Trust and the National Institute for Health Research Biomedical Research Centre at University College London Hospitals NHS Foundation Trust, the project aims to contribute towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods.

### Installation
The following are prerequisites:   
  
*itk-snap  
Python3  
python3 -m pip install numpy  
python3 -m pip install nibabel  
sudo apt install python3-tk (alternatively, sudo yum install)  
python3 -m pip install matplotlib  
python3 -m pip install pandas  
python3 -m pip install joblib  
python3 -m pip install scikit-learn  
python3 -m pip install tensorflow  
python3 -m pip install keras  
python3 -m pip install scikit-image*  

### Acknowledgements
The software includes icons by [Icon8](https://icons8.com)

### Starting Musclesense Workbench
`python3 mmseg_app.py` to run

### Troubleshooting
1.  `assert(data['image_dim_ordering']=="th") KeyError: 'image_dim_ordering'`
Add `"image_dim_ordering": "th"` to `~/.keras/keras.json`
