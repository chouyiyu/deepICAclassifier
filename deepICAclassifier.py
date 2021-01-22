import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os,sys

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import nibabel as nib
from tensorflow.keras.backend import eval
import argparse
from keras.models import model_from_json
from urllib.request import urlopen

def pairwise_distance(feature, squared=False):
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):

    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):

    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def create_base_network(image_input_shape, embedding_size):
    input_image = Input(shape=image_input_shape)

    x = Flatten()(input_image)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(embedding_size)(x)

    base_network = Model(inputs=input_image, outputs=x)

    return base_network


def normalize(img,thr):
    ind = np.where(img <thr)
    img[ind]=0
    # normalize inputs to 0-1
    max_v = np.percentile(img,99.5)
    img[img>max_v]=max_v
    min_v = 0
    img = (img - min_v)  / (max_v - min_v)

    return img

def dist(model,embedding_size,imgA,imgB):
    input_image_shape = (40, 48, 38, 1)

    # creating an empty network
    testing_embeddings = create_base_network(input_image_shape,
                                             embedding_size=embedding_size)

    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

    test=np.zeros((2, 40,48,38))

    img0=nib.load(imgA)
    img0_data=normalize(img0.get_data(),2.3)
    img1=nib.load(imgB)
    img1_data=normalize(img1.get_data(),2.3)

    test[0,:,:,:]=img0_data
    test[1,:,:,:]=img1_data

    x_embeddings = testing_embeddings.predict(np.reshape(test, (len(test), 40, 48, 38, 1)))
 
    pdist = pairwise_distance(x_embeddings, squared=True)
    distM=eval(pdist)
 
    dist=(distM[:,-1])
    print("img1:{}".format(imgA))
    print("img2:{}".format(imgB))    
    print("Distance:{}".format(dist[:-1]))

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode",type=str, default='train')
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--epochs", type=int, default=1000)	
	parser.add_argument("--embedding_size", type=int, default=64)
	parser.add_argument("--traindir", type=str,default="/ISFILE3/USERS/chouy/Projects/DeepLearning/SiameseICA/v10/trainx10")
	parser.add_argument("--testdir", type=str,default="/ISFILE3/USERS/chouy/Projects/DeepLearning/SiameseICA/v10/testingx10")
	parser.add_argument("--img1", type=str)
	parser.add_argument("--img2", type=str)
	parser.add_argument("--img4d", type=str)
	parser.add_argument("--icadir", type=str)	
	parser.add_argument("--gpu", type=str,default='0')
	parser.add_argument("--model", type=str)
	parser.add_argument("--png", type=str)	
	parser.add_argument("--lrate", type=float,default=0.0001)

	args=parser.parse_args()

	return args

# classification of the ICA component 
def classify(my_model,embedding_size,img):
 
    input_image_shape = (40, 48, 38, 1)

    # creating an empty network
    testing_embeddings = create_base_network(input_image_shape,
                                             embedding_size=embedding_size)
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, my_model.layers[2].layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

    test=np.zeros((12, 40,48,38))
    img0=nib.load('template/dm.nii.gz')
    img0_data=normalize(img0.get_data(),2.3)
    img1=nib.load('template/visM.nii.gz')
    img1_data=normalize(img1.get_data(),2.3)
    img2=nib.load('template/visO.nii.gz')
    img2_data=normalize(img2.get_data(),2.3)
    img3=nib.load('template/visL.nii.gz')
    img3_data=normalize(img3.get_data(),2.3)
    img4=nib.load('template/motor.nii.gz')
    img4_data=normalize(img4.get_data(),2.3)
    img5=nib.load('template/auditory.nii.gz')
    img5_data=normalize(img5.get_data(),2.3)
    img6=nib.load('template/cerebellum.nii.gz')
    img6_data=normalize(img6.get_data(),2.3)
    img7=nib.load('template/executive.nii.gz')
    img7_data=normalize(img7.get_data(),2.3)
    img8=nib.load('template/salience.nii.gz')
    img8_data=normalize(img8.get_data(),2.3)
    img9=nib.load('template/danL.nii.gz')
    img9_data=normalize(img9.get_data(),2.3)
    img10=nib.load('template/danR.nii.gz')
    img10_data=normalize(img10.get_data(),2.3)

    test[0,:,:,:]=img0_data
    test[1,:,:,:]=img1_data
    test[2,:,:,:]=img2_data
    test[3,:,:,:]=img3_data
    test[4,:,:,:]=img4_data
    test[5,:,:,:]=img5_data
    test[6,:,:,:]=img6_data
    test[7,:,:,:]=img7_data
    test[8,:,:,:]=img8_data
    test[9,:,:,:]=img9_data
    test[10,:,:,:]=img10_data
   
    img11=nib.load(img)
    img11_data=normalize(img11.get_data(),2.3)

    test[11,:,:,:]=img11_data

    x_embeddings = testing_embeddings.predict(np.reshape(test, (len(test), 40, 48, 38, 1)))
 
    pdist = pairwise_distance(x_embeddings)
    distM=eval(pdist)
    dist=(distM[:,-1])
 
    index=np.argmin(dist[:-1])
    if index==0:
        net='Default Mode'
    elif index==1:
        net='Medial Visual Network'
    elif index==2:
        net='Occipital Visual Network'
    elif index==3:
        net='Lateral Visual Network'
    elif index==4:
        net='Motor Network'
    elif index==5:
        net='Auditory Network'
    elif index==6:
        net='Cerebellum Network'
    elif index==7:
        net='Executive Network'        
    elif index==8:
        net='Salience Network'
    elif index==9:
        net='L Dorsal Attentation Network'
    elif index==10:
        net='R Dorsal Attentation Network'

    print("ICA: {} -- {}".format(img,net))        

# label the best ICA components 
def bestICA(my_model,embedding_size,img):
 
    input_image_shape = (40, 48, 38, 1)

    # creating an empty network
    testing_embeddings = create_base_network(input_image_shape,
                                             embedding_size=embedding_size)
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, my_model.layers[2].layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

    test=np.zeros((12, 40,48,38))
    img0=nib.load('template/dm.nii.gz')
    img0_data=normalize(img0.get_data(),2.3)
    img1=nib.load('template/visM.nii.gz')
    img1_data=normalize(img1.get_data(),2.3)
    img2=nib.load('template/visO.nii.gz')
    img2_data=normalize(img2.get_data(),2.3)
    img3=nib.load('template/visL.nii.gz')
    img3_data=normalize(img3.get_data(),2.3)
    img4=nib.load('template/motor.nii.gz')
    img4_data=normalize(img4.get_data(),2.3)
    img5=nib.load('template/auditory.nii.gz')
    img5_data=normalize(img5.get_data(),2.3)
    img6=nib.load('template/cerebellum.nii.gz')
    img6_data=normalize(img6.get_data(),2.3)
    img7=nib.load('template/executive.nii.gz')
    img7_data=normalize(img7.get_data(),2.3)
    img8=nib.load('template/salience.nii.gz')
    img8_data=normalize(img8.get_data(),2.3)
    img9=nib.load('template/danL.nii.gz')
    img9_data=normalize(img9.get_data(),2.3)
    img10=nib.load('template/danR.nii.gz')
    img10_data=normalize(img10.get_data(),2.3)

    test[0,:,:,:]=img0_data
    test[1,:,:,:]=img1_data
    test[2,:,:,:]=img2_data
    test[3,:,:,:]=img3_data
    test[4,:,:,:]=img4_data
    test[5,:,:,:]=img5_data
    test[6,:,:,:]=img6_data
    test[7,:,:,:]=img7_data
    test[8,:,:,:]=img8_data
    test[9,:,:,:]=img9_data
    test[10,:,:,:]=img10_data
   
    img11=nib.load(img)
    img11_data=img11.get_data()
    scores=np.ones((img11_data.shape[3],11))*100000

    for i in range(img11_data.shape[3]):
        tmp=normalize(np.squeeze(img11_data[:,:,:,i]),2.3)
        test[11,:,:,:]=tmp
        x_embeddings = testing_embeddings.predict(np.reshape(test, (len(test), 40, 48, 38, 1)))
 
        pdist = pairwise_distance(x_embeddings)
        distM=eval(pdist)
        dist=(distM[:,-1])
        scores[i,:]=dist[:-1]
 
    for index in range(11):
        ind=np.unravel_index(np.argmin(scores[:,index],axis=None),scores[:,0].shape)

        if index==0:
            net='Default Mode Network'
        elif index==1:
            net='Medial Visual Network'
        elif index==2:
            net='Occipital Visual Network'
        elif index==3:
            net='Lateral Visual Network'
        elif index==4:
            net='Motor Network'
        elif index==5:
            net='Auditory Network'
        elif index==6:
            net='Cerebellum Network'
        elif index==7:
            net='Executive Network'        
        elif index==8:
            net='Salience Network'
        elif index==9:
            net='L Dorsal Attentation Network'
        elif index==10:
            net='R Dorsal Attentation Network'

        print("Best {} - [{}]".format(net,ind[0]))    
       
if __name__ == "__main__":
	args=get_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	# load json and create model
	json_file = open('model/ica_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	if not os.path.isfile('model/ica_weight.h5'):
		url ="https://zenodo.org/record/4458279/files/ica_weight.h5?download=1"

		print("Downloading", url, "...")
		data = urlopen(url).read()
		with open('model/ica_weight.h5', 'wb') as f:
			f.write(data)

	# load weights into new model
	loaded_model.load_weights("model/ica_weight.h5")
	print("Loaded model from disk")

	embedding_size =64

	if args.mode=='dist':
		dist(loaded_model,embedding_size,args.img1,args.img2)
	elif args.mode=='classify':
		# classify the ICA component
		classify(loaded_model,embedding_size,args.img1)
	elif args.mode=='bestICA':
		# label the best ICA components
		bestICA(loaded_model,embedding_size,args.img1)

