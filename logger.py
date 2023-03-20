from pytorch_lightning.callbacks import Callback
import torchvision
import torch

class BrainNetImageLogger(Callback):
    def __init__(self,num_features = 3 , num_images=12, log_steps=10,mean=0,std=0.015):
        self.num_features = num_features
        self.log_steps = log_steps
        self.num_images = num_images
        self.mean = mean
        self.std = std

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

            VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = batch

            VL = VL.to(pl_module.device,non_blocking=True)
            FL = FL.to(pl_module.device,non_blocking=True)
            VFL = VFL.to(pl_module.device,non_blocking=True)
            FFL = FFL.to(pl_module.device,non_blocking=True)
            VR = VR.to(pl_module.device,non_blocking=True)
            FR = FR.to(pl_module.device,non_blocking=True)
            VFR = VFR.to(pl_module.device,non_blocking=True)
            FFR = FFR.to(pl_module.device,non_blocking=True)
            demographic = demographic.to(pl_module.device,non_blocking=True)

            with torch.no_grad():

                images, PF = pl_module.render(VL, FL, VFL, FFL)                    

                grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image features', grid_images, pl_module.global_step)

                images_noiseM = pl_module.noise(images)

                grid_images_noiseM = torchvision.utils.make_grid(images_noiseM[0, 0:self.num_images, 0:self.num_features, :, :])
                trainer.logger.experiment.add_image('Image + noise M ', grid_images_noiseM, pl_module.global_step)