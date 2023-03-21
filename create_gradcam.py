#CUDA_VISIBLE_DEVICES=0

import os 

import numpy as np
import cv2

import torch
from torch import nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

import monai
import pandas as pd


from net import BrainIcoNet
from data import BrainIBISDataModule

from transformation import RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform


import numpy as np
import random
import torch
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

import plotly.express as pd

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene

class Classification_for_gradcam(nn.Module):
    def __init__(self,classification_layer,xR,demographic):
        super().__init__()
        self.classification_layer = classification_layer
        self.xR = xR
        self.demographic = demographic

 
    def forward(self,x):
        l = [x,self.xR,self.demographic]
        x = torch.cat(l,dim=1)
        
        x = self.classification_layer(x)
        
        return x


# nbr_fold = 1
# nbr_p = 5 
# support = 'sphere' #sphere,'flatten_brain','inflated_brain'
# kind_of_layer = "Ico" #"Att","Ico"
# kind_of_dataset = "LR" #"","LR"

##############################################################################################Hyperparamters
batch_size = 10
num_workers = 12 
image_size = 224
noise_lvl = 0.03
dropout_lvl = 0.2
ico_lvl = 1
radius = 2  
lr = 1e-4

#parameters for GaussianNoiseTransform
mean = 0
std = 0.005

###Transformation
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

###Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###Path
path_data = "/MEDUSA_STOR/ugor/IBIS_sa_eacsf_thickness"
data_train = "dataASDdemographicsLR-V06_12fold0_train.csv"
data_val = "dataASDdemographicsLR-V06_12fold0_val.csv"
data_test = "dataASDdemographicsLR-V06_12fold0_test.csv"
path_ico_left = 'sphere_f327680_v163842.vtk'
path_ico_right = 'sphere_f327680_v163842.vtk'
list_path_ico = [path_ico_left,path_ico_right]


###Demographics
list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR

###Resampling
resampling = 'resampling_ASD' #'no_resampling','resampling_no_ASD','resampling_ASD'
#Choose between these 3 choices to balence your data. Per default : no_resampling.

###IcoLayer
IcoLayer = 'IcoConv2D' #'IcoConv2D','IcoConv1D','IcoLinear'
#Choose between these 3 choices to choose what kind of IcoLayer we want to use

###Name model
name_model = '/work2/ugor/brain_classification/Classification_ASD/Checkpoint/TILRP5IcoF0/epoch=5-val_loss=0.68.ckpt'

##############################################################################################Hyperparamters




brain_data = BrainIBISDataModule(batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,resampling=resampling,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)#MLR
nbr_features = brain_data.get_features()
nbr_demographic = brain_data.get_nbr_demographic()
weights = brain_data.get_weigths()
brain_data.setup() 
nbr_brain = brain_data.test_dataset.__len__()


model = BrainIcoNet(IcoLayer,nbr_features,nbr_demographic,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr)

checkpoint = torch.load(name_model, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

list_gradcam_L = []
list_gradcam_R = []

classification_layer = model.Classification

n_targ = 1
targets = [ClassifierOutputTarget(n_targ)]

for j in range(nbr_brain): #nbr_brain
    print("j : ",j)
    VL, FL, VFL, FFL,VR, FR, VFR, FFR, demographic, Y= brain_data.test_dataset.__getitem__(j)
    VL = VL.unsqueeze(dim=0).to(device)
    FL = FL.unsqueeze(dim=0).to(device)
    VFL = VFL.unsqueeze(dim=0).to(device)
    FFL = FFL.unsqueeze(dim=0).to(device)
    VR = VR.unsqueeze(dim=0).to(device)
    FR = FR.unsqueeze(dim=0).to(device)
    VFR = VFR.unsqueeze(dim=0).to(device)
    FFR = FFR.unsqueeze(dim=0).to(device)
    demographic = demographic.unsqueeze(dim=0).to(device)

    xL, PF = model.render(VL,FL,VFL,FFL)
    xR, PF = model.render(VR,FR,VFR,FFR)

    featuresL = model.poolingL(model.IcosahedronConv2dL(model.TimeDistributedL(xL))) #LORR
    featuresR = model.poolingR(model.IcosahedronConv2dR(model.TimeDistributedL(xR))) 

    classifier_for_gradcamL = Classification_for_gradcam(classification_layer,featuresR, demographic)#LORR
    classifier_for_gradcamR = Classification_for_gradcam(classification_layer,featuresL, demographic)    

    ##### I initialize the gradcam
    model_camL = nn.Sequential(model.TimeDistributedL, model.IcosahedronConv2dL, model.poolingL,classifier_for_gradcamL)
    model_camR = nn.Sequential(model.TimeDistributedR, model.IcosahedronConv2dR, model.poolingR,classifier_for_gradcamR)#LORR
    target_layersL = [model_camL[0].module.layer4[-1]]
    target_layersR = [model_camR[0].module.layer4[-1]]

    camL = GradCAM(model=model_camL, target_layers=target_layersL)
    camR = GradCAM(model=model_camR, target_layers=target_layersR)

    grayscale_camL = torch.Tensor(camL(input_tensor=xL, targets=targets))
    grayscale_camR = torch.Tensor(camR(input_tensor=xR, targets=targets))

    list_gradcam_L.append(grayscale_camL.unsqueeze(dim=1))
    list_gradcam_R.append(grayscale_camR.unsqueeze(dim=1))

t_gradcam_L = torch.cat(list_gradcam_L,dim=1)
t_gradcam_R = torch.cat(list_gradcam_R,dim=1)
tL = torch.mean(t_gradcam_L,dim=1)
tR = torch.mean(t_gradcam_R,dim=1)

torch.save(tL,'Tensor/left_gradcam'+'.pt')
torch.save(tR,'Tensor/right_gradcam'+'.pt')

