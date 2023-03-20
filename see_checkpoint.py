import os

import numpy as np
import cv2

import torch
from torch import nn
from torchvision.models import resnet50

import monai
import pandas as pd

from net import BrainIcoNet
from data import BrainIBISDataModule

from transformation import RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform


import numpy as np
import random
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 

from vtk.util.numpy_support import vtk_to_numpy
import vtk

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd

##############################################################################################
batch_size = 10
num_workers = 12 #6-12
image_size = 224
noise_lvl = 0.03
dropout_lvl = 0.2
num_epochs = 100
ico_lvl = 1
radius = 2 
lr = 1e-4

mean = 0
std = 0.01

path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"
data_train = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDdemographicsLR-V06_12fold0_train.csv"
data_val = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDdemographicsLR-V06_12fold0_val.csv"
data_test = "/NIRAL/work/ugor/source/brain_classification/Classification_ASD/Data/dataASDdemographicsLR-V06_12fold0_test.csv"
path_ico_left = '/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk'
path_ico_right = '/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk'
list_path_ico = [path_ico_left,path_ico_right]

#Transform
list_train_transform = []    
list_train_transform.append(CenterTransform())
list_train_transform.append(NormalizePointTransform())
list_train_transform.append(RandomRotationTransform())
list_train_transform.append(GaussianNoisePointTransform(mean,std))
list_train_transform.append(NormalizePointTransform())

train_transform = monai.transforms.Compose(list_train_transform)

list_val_and_test_transform = []    
list_val_and_test_transform.append(CenterTransform())
list_val_and_test_transform.append(NormalizePointTransform())

val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)


path_model = 'Checkpoint/something/something'


list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR
##############################################################################################


brain_data = BrainIBISDataModule(batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)#MLR
nbr_features = brain_data.get_features()
nbr_demographic = brain_data.get_nbr_demographic()
weights = brain_data.get_weigths()

model = BrainIcoNet(nbr_features,nbr_demographic,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr)#MLR
checkpoint = torch.load(path_model)
#checkpoint = torch.load(path_model,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

trainer = Trainer(max_epochs=num_epochs,accelerator="gpu")
#trainer = Trainer(max_epochs=num_epochs)

trainer.test(model, datamodule=brain_data)