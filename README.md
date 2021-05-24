*Musclesense Workbench* is a software application built around *Musclesense*, introduced by the following article:

*[Musclesense: a Trained, Artificial Neural Network for the Anatomical Segmentation of Lower Limb Magnetic Resonance Images in Neuromuscular Diseases, Neuroinformatics. 2021 Apr;19(2):379-383.](https://pubmed.ncbi.nlm.nih.gov/32892313/)*

A pure python3 application, it has been designed to support the processing and analysis of 3-point Dixon MRI of the lower limbs. 

The software was produced as part of the project ***“Towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods”***. 

Funded by the Wellcome Trust and the National Institute for Health Research Biomedical Research Centre at University College London Hospitals NHS Foundation Trust, the project aims to contribute towards improving the clinical care of patients with neuromuscular diseases using innovative artificial intelligence imaging methods.

![image](https://user-images.githubusercontent.com/12815964/119245568-31236080-bb72-11eb-8e33-fccad5aab35f.png)

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

Includes icons by [Icon8](https://icons8.com)

`python3 mmseg_app.py` to run

### Troubleshooting
1.  `assert(data['image_dim_ordering']=="th") KeyError: 'image_dim_ordering'`
Add `"image_dim_ordering": "th"` to `~/.keras/keras.json`
