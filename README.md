# deepICAclassifier
automatically classification of the components derived from ICA analysis of rsfMRI data as particular functional brain resting state networks. Currently, the available functional networks are Default Mode Network, Lateral Visual Network, Occipital Visual Network, Lateral Visual Network, Motor Network, Auditory Network, Cerebellum Network, Executive Network, Salience Network, Left Dorsal Attentation Network and Right Dorsal Attentation Network.
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
option 2: label the best ICA components for each funcational brain resting state network. The output is the index (starting from 0) of the 4th dimentation for a 4D ICA input data.
```
python3 deepICAclassifier.py --mode bestICA --img1 /path/to/4d_ica --gpu 0

```
option 3: compute the distance (similarity) between image1 and image2 in the embedding space
```
python3 deepICAclassifier.py --mode dist --img1 /path/to/image1 --img2 /path/to/image2 --gpu 0 
```
Input images must be in NIfTI format and rigidly registered to the MNI space (template provided as MNI_template.nii) with dimension 40x48x38 and voxel size 4mm^3. By default, deepImgContrast will run on CPU mode, set the option --gpu to 0,1,2 .... for running on GPU mode. 
#
