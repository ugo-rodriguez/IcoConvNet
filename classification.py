import numpy as np
import torch
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import monai
import nibabel as nib


from net import BrainIcoNet
from data import BrainIBISDataModule
from logger import BrainNetImageLogger

from transformation import RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform


print("Import // done")

def main():

    # nbr_fold = 0
    # nbr_p = 5
    # support = 'sphere' #sphere,'flatten_brain','inflated_brain'
    # kind_of_layer = "Ico" #"Att","Ico"
    # kind_of_dataset = "LR" #"","LR"

    ##############################################################################################Hyperparamters
    batch_size = 10 
    num_workers = 12 
    image_size = 224
    noise_lvl = 0.01
    dropout_lvl = 0.2
    num_epochs = 150
    ico_lvl = 1
    radius = 2 
    lr = 1e-4

    #parameters for GaussianNoiseTransform
    mean = 0
    std = 0.005

    #parameters for EarlyStopping
    min_delta_early_stopping = 0.00
    patience_early_stopping = 30

    #Paths
    path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"
    data_train = "dataASDdemographicsLR-V06_12fold0_train.csv"
    data_val = "dataASDdemographicsLR-V06_12fold0_val.csv"
    data_test = "dataASDdemographicsLR-V06_12fold0_test.csv"
    path_ico_left = 'sphere_f327680_v163842.vtk'
    path_ico_right = 'sphere_f327680_v163842.vtk'
    list_path_ico = [path_ico_left,path_ico_right]


    name = 'IcoLRF0' #Choose the name of your name
    print('name to save checkpoints : ',name)

    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR

    ###Transformation
    list_train_transform = [] 
    list_train_transform.append(CenterTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())   
    list_train_transform.append(GaussianNoisePointTransform(mean,std)) # Don't use this transformation if your object isn't a sphere
    list_train_transform.append(NormalizePointTransform()) # Don't use this transformation if your object isn't a sphere

    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterTransform())
    list_val_and_test_transform.append(NormalizePointTransform())

    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

    ###Resampling
    resampling = 'resampling_ASD' #'no_resampling','resampling_no_ASD','resampling_ASD'
    #Choose between these 3 choices to balence your data. Per default : no_resampling.

    ###IcoLayer
    IcoLayer = 'IcoConv2D' #'IcoConv2D','IcoConv1D','IcoLinear'
    #Choose between these 3 choices to choose what kind of IcoLayer we want to use

    ##############################################################################################







    #Get number of images
    list_nb_verts_ico = [12,42]
    nb_images = list_nb_verts_ico[ico_lvl-1]
    
    #Creation of Dataset
    brain_data = BrainIBISDataModule(batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,resampling=resampling,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)#MLR
    nbr_features = brain_data.get_features()
    nbr_demographic = brain_data.get_nbr_demographic()
    weights = brain_data.get_weigths()

    #Creation of our model
    model = BrainIcoNet(IcoLayer,nbr_features,nbr_demographic,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, weights,radius=radius,lr=lr,name=name)#MLR

    #Creation of Checkpoint (if we want to save best models)
    checkpoint_callback = ModelCheckpoint(
        dirpath=name,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    ) #To save best models according to validation loss  

    #Logger (Useful if we use Tensorboard)
    logger = TensorBoardLogger(save_dir="test_tensorboard", name="my_model")

    #Early Stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode="min")

    #Image Logger (Useful if we use Tensorboard)
    image_logger = BrainNetImageLogger(num_features = nbr_features,num_images = nb_images,mean = 0,std=noise_lvl)



    ###Trainer
    trainer = Trainer(log_every_n_steps=10,reload_dataloaders_every_n_epochs=True,logger=logger,max_epochs=num_epochs,callbacks=[early_stop_callback,checkpoint_callback,image_logger],accelerator="gpu") 

    trainer.fit(model,datamodule=brain_data)

    trainer.test(model, datamodule=brain_data)

if __name__ == '__main__':
    main()
