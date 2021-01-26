# deepICAclassifier
Automatic classification of the components derived from ICA analysis of rsfMRI data into functional brain resting state networks. Currently, the available functional networks are Default Mode Network, Medial Visual Network, Occipital Visual Network, Lateral Visual Network, Motor Network, Auditory Network, Cerebellum Network, Executive Network, Salience Network, Left Dorsal Attentation Network and Right Dorsal Attentation Network.
## Installation
```
git clone https://github.com/chouyiyu/deepICAclassifier.git
```
## Prerequisites
```
python3
keras
tensorflow
numpy
nibabel
```
## How to use it
option 1: classify the ICA component as one of the funcational brain resting state network by finding the shortest distance among the support set provided under /template.  
```
python3 deepICAclassifier.py --mode classify --img1 /path/to/3d_ica --gpu 0

```
option 2: label the best-fit component for each funcational brain resting state network of a 4D ICA image data. The output is the index (starting from 0) of the 4th dimension.
```
python3 deepICAclassifier.py --mode bestICA --img1 /path/to/4d_ica --gpu 0

```
option 3: compute the distance (similarity) between image1 and image2 in the embedding space
```
python3 deepICAclassifier.py --mode dist --img1 /path/to/3d_ica1 --img2 /path/to/3d_ica2 --gpu 0 
```
Inputs are the Z-Score images computed by FSL melodic ICA software with dimension 40x48x38, voxel size 4mm^3 and registered to the TT_N27 space (template provided as TT_N27_dxyz4_brain.nii). By default, deepICAclassifier will run on CPU mode, set the option --gpu to 0,1,2 .... for running on GPU mode. 
#
