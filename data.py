import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 
from torch.nn.functional import normalize

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd


class BrainIBISDataset(Dataset):
    def __init__(self,df,list_demographic,path_data,list_path_ico,transform = None,version=None, column_subject_id='Subject_ID', column_age='Age',column_hemisphere = 'Hemisphere',column_ASD = 'ASD_administered'):
        self.df = df
        self.list_demographic = list_demographic
        self.path_data = path_data
        self.list_path_ico = list_path_ico
        self.transform = transform
        self.version = version
        self.column_subject_id =column_subject_id
        self.column_age = column_age
        self.column_hemisphere = column_hemisphere
        self.column_ASD = column_ASD

    def __len__(self):
        return(len(self.df)) 

    def __getitem__(self,idx):

        row = self.df.loc[idx]

        #Get item for each hemisphere (left and right)
        vertsL, facesL, vertex_featuresL, face_featuresL, Y = self.getitem_per_hemisphere('left', idx)
        vertsR, facesR, vertex_featuresR, face_featuresR, Y = self.getitem_per_hemisphere('right', idx)

        #Get demographics
        demographic_values = [float(row[name]) for name in self.list_demographic]
        demographic = torch.tensor(demographic_values)

        return  vertsL, facesL, vertex_featuresL, face_featuresL, vertsR, facesR, vertex_featuresR, face_featuresR, demographic, Y
    
    def getitem_per_hemisphere(self,hemisphere,idx):
        #Load Data
        row = self.df.loc[idx]
        number_brain = str(int(row[self.column_subject_id]))

        l_version = ['V06','V12']
        idx_version = int(row[self.column_age])
        version = l_version[idx_version]

        idx_ASD = int(row[self.column_ASD])

        l_features = []

        path_eacsf = f"{self.path_data}/{number_brain}/{version}/eacsf/{hemisphere}_eacsf.txt"
        path_sa =    f"{self.path_data}/{number_brain}/{version}/sa/{hemisphere}_sa.txt"
        path_thickness = f"{self.path_data}/{number_brain}/{version}/thickness/{hemisphere}_thickness.txt"

        eacsf = open(path_eacsf,"r").read().splitlines()
        eacsf = torch.tensor([float(ele) for ele in eacsf])
        l_features.append(eacsf.unsqueeze(dim=1))

        sa = open(path_sa,"r").read().splitlines()
        sa = torch.tensor([float(ele) for ele in sa])
        l_features.append(sa.unsqueeze(dim=1))

        thickness = open(path_thickness,"r").read().splitlines()
        thickness = torch.tensor([float(ele) for ele in thickness])
        l_features.append(thickness.unsqueeze(dim=1))

        vertex_features = torch.cat(l_features,dim=1)

        Y = torch.tensor([idx_ASD])

        #Load  Icosahedron
        if hemisphere == 'left':
            reader = utils.ReadSurf(self.list_path_ico[0])
        else:
            reader = utils.ReadSurf(self.list_path_ico[1])
        verts, faces, edges = utils.PolyDataToTensors(reader)

        nb_faces = len(faces)

        #Transformations
        if self.transform:        
            verts = self.transform(verts)

        #Face Features
        faces_pid0 = faces[:,0:1]         
    
        offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])      
        
        face_features = torch.take(vertex_features,faces_pid0_offset)

        return verts, faces,vertex_features,face_features, Y

class BrainIBISDataModule(pl.LightningDataModule):
    def __init__(self,batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,resampling='no_resampling',train_transform=None,val_and_test_transform=None, num_workers=6, pin_memory=False, persistent_workers=False):
        super().__init__()
        self.batch_size = batch_size 
        self.list_demographic = list_demographic
        self.path_data = path_data
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.list_path_ico = list_path_ico
        self.resampling = resampling
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers=persistent_workers

        ### weights computing
        self.weights = []

        self.df_train = pd.read_csv(self.data_train)
        self.df_val = pd.read_csv(self.data_val)
        self.df_test = pd.read_csv(self.data_test)

        y_train = np.array(self.df_train.loc[:,'ASD_administered'])
        labels = np.unique(y_train)
        class_weights_train  = torch.tensor(class_weight.compute_class_weight('balanced',classes=labels,y=y_train)).to(torch.float32) 
        self.weights.append(class_weights_train) 

        y_val = np.array(self.df_val.loc[:,'ASD_administered'])
        class_weights_val = torch.tensor(class_weight.compute_class_weight('balanced',classes=labels,y=y_val)).to(torch.float32)
        self.weights.append(class_weights_val) 

        y_test = np.array(self.df_test.loc[:,'ASD_administered'])
        class_weights_test = torch.tensor(class_weight.compute_class_weight('balanced',classes=labels,y=y_test)).to(torch.float32)
        self.weights.append(class_weights_test) 

        self.setup()


    def setup(self,stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_dataset = BrainIBISDataset(self.df_train,self.list_demographic,self.path_data,self.list_path_ico,self.train_transform)
        self.val_dataset = BrainIBISDataset(self.df_val,self.list_demographic,self.path_data,self.list_path_ico,self.val_and_test_transform)
        self.test_dataset = BrainIBISDataset(self.df_test,self.list_demographic,self.path_data,self.list_path_ico,self.val_and_test_transform)

        VL, FL, VFL, FFL,VR, FR, VFR, FFR, demographic, Y = self.train_dataset.__getitem__(0)
        self.nbr_features = VL.shape[1]
        self.nbr_demographic = demographic.shape[0]

    def train_dataloader(self):
        if self.resampling == 'resampling_no_ASD':
            df_healthy = self.df_train.query('ASD_administered == 0')
            df_ASD = self.df_train.query('ASD_administered == 1')
            nbr_ASD = len(df_ASD)
            df_healthy = df_healthy.sample(nbr_ASD)
            new_df_train = pd.concat([df_healthy,df_ASD]).reset_index()
            self.train_dataset = BrainIBISDataset(new_df_train,self.list_demographic,self.path_data,self.list_path_ico,self.train_transform)
            
        elif self.resampling == 'resampling_ASD':
            df_healthy = self.df_train.query('ASD_administered == 0')
            df_ASD = self.df_train.query('ASD_administered == 1')
            nbr_healthy = len(df_healthy)
            df_ASD = self.repeat_subject(df_ASD,nbr_healthy)
            new_df_train = pd.concat([df_healthy,df_ASD]).reset_index().drop(['index'],axis=1)
            self.train_dataset = BrainIBISDataset(new_df_train,self.list_demographic,self.path_data,self.list_path_ico,self.train_transform)

        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def repeat_subject(self,df,final_size):
        n = len(df)
        q,r = final_size//n,final_size%n
        list_df = [df for i in range(q)]
        list_df.append(df[:r])
        new_df = pd.concat(list_df).reset_index().drop(['index'],axis=1)
        return new_df

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)        

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def get_features(self):
        return self.nbr_features

    def get_weigths(self):
        return self.weights

    def get_nbr_demographic(self):
        return self.nbr_demographic