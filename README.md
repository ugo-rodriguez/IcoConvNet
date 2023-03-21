(PS : my english is maybe approximate, sorry for that =) )


# How to use this project ?
classification.py is the main of my project. To use BrainIcoConv, the method created by us, you have to run classification.py. We are going to see and explain the lines that you can change or in this project.

in classification.py : 

- path_data : path_data is the path where we can find the data. Change with your own data.

- data_train / data_val / data_test : Files where we can find information on each subject. The first three columns ('Subject_ID,Age,ASD_administered') are required to use BrainIcoConv. The others are optional, used during demographics. Subject_ID corresponds to the ID of each subject. Age corresponds to the age of the brain (0 = 6 months, 1 = 12 months). ASD_administered corresponds to the classes of my classification (0 = No ASD, 1 = ASD). Change with your own data. 

- path_ico_left / path_ico_right : 3D vtk Object correponds to the shape of your subjects. For the example, I give Icosphere (subdivision level = 6) but it can be any 3D shape. 

- list_demographic : use your own demographics in data_train, data_val or data_test.

- Transformation lines : These lines are transforms used on the 3D Object. The order of transforms has an importance. Transforms during train step and val/test step aren't the same. Don't use last two transforms for train step if your object isn't a sphere.

- resampling : choose between these 3 choices to balence your data. Per default : no_resampling.

- IcoLayer : choose between these 3 choices to choose what kind of IcoLayer we want to use

- checkpoint_callback : if you want to save the best model.

- logger / image_logger : if you want to use Tensorboard.


in data.py : 

- getitem : getitem allows to extract information of each subject. 1 subject is 1 brain with 2 hemispheres (left and right). In this case, Each hemisphere has 3 features (eacsf, sa, thickness). You need maybe to change for your own data and your own features. The ouputs of getitem are 'verts', 'faces', 'vertex_features', 'face_features' tensors for each hemisphere, 'demographics' tensor and 'Y' for the class. You can maybe change the output if you don't need to load like 2 3D Objects for each subject.

in net.py :

- train_step / val_step / test_step / forward / init / get_features : if you don't load 2 3D Objects for each subject, you need to change some lines and remove 1 path (left or right path) to be sure that this code works with your data.










# What is see_checkpoint.py ?

This python file is just to read 1 saved checkpoint to see again the classification report. 

