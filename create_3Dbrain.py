import numpy as np

import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import utils 

##############################################################################################Hyperparamters
path_brain = '3D_vtk_Object/icbm_avg_white_sym_mc_right_hires_flatten.vtk'
#path_brain = '3D_vtk_Object/icbm_avg_mid_sym_mc_right_hires.vtk'

name_PF = 'Tensor/PF.pt'

name_gradcam = 'Tensor/left_gradcam.pt'

saved_name = 'brain'
##############################################################################################Hyperparamters

surface = utils.ReadSurf(path_brain)
verts, faces, edges = utils.PolyDataToTensors(surface)

nbr_verts = verts.shape[0]
nbr_faces = faces.shape[0]

Colors = vtk.vtkDoubleArray()
Colors.SetNumberOfComponents(1)
Colors.SetName('Colors')

PF = torch.load(name_PF, map_location=torch.device('cpu')).squeeze(dim=2)[0]
num_views = PF.shape[0]
image_size = PF.shape[1]
gradcam = torch.load(name_gradcam, map_location=torch.device('cpu'))

gradcam_points = torch.zeros(nbr_verts,3)
intermediaire_gradcam_faces = torch.zeros(nbr_faces)
gradcam_faces = torch.zeros(nbr_faces)
gradcam_count = torch.zeros(nbr_faces)

for cam in range(num_views):

    reshape_size = [image_size*image_size]

    PF_image = PF[cam]
    PF_image = PF_image.contiguous().view(reshape_size)

    gradcam_image = gradcam[cam]
    gradcam_image = gradcam_image.contiguous().view(reshape_size)

    intermediaire_gradcam_faces[PF_image] = gradcam_image

    gradcam_faces += intermediaire_gradcam_faces
    gradcam_count[PF_image] += 1.0

    intermediaire_gradcam_faces = torch.zeros(nbr_faces)


print(2)
zeros_count = ((gradcam_count == 0).nonzero(as_tuple=True)[0])
gradcam_count[zeros_count] = torch.ones(zeros_count.shape[0])
gradcam_faces = torch.div(gradcam_faces,gradcam_count)
gradcam_faces[-1] = gradcam_faces[-2]



for i in range(3):
    ID_verts = faces[:,i]
    gradcam_points[:,i][ID_verts.long()] = gradcam_faces
gradcam_points = torch.max(gradcam_points,dim=1)[0].to(torch.double)
print(gradcam_points)


Colors = vtk.vtkDoubleArray()
Colors.SetNumberOfComponents(1)
Colors.SetName('Colors')

for c in range(nbr_verts):
    Colors.InsertNextTypedTuple([gradcam_points[c].item()])
surface.GetPointData().SetScalars(Colors)

writer = vtk.vtkPolyDataWriter()
writer.SetInputData(surface)
title = saved_name+'.vtk'
writer.SetFileName(title)
writer.Update()
#writer.Write()



print(2)