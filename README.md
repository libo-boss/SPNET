# SPNET
In this study, an end-to-end network (SP-Net) was developed to extract features from Biplane Radiographs to reconstruct the 3D structure of the spine. We also explored multiple configurations of the SP-Net in order to improve the accuracy of the 3D reconstruction of the spine. We defined evaluation metrics to compare with several classical 3D model evaluation methods.


Required environment:

python 3.6

keras 2.6.0

tensorflow 2.6.2


Dataset: 

Under the SP-net/dataset folder, there are two datasets, one for normal spine (normal) and one for scoliosis spine (scoliosis). We used 8 datasets in our paper, this is only a part of the data and we will continue to upload other data.

Dataset production: 

If you have CT data that is licensed for use, you can produce the same dataset as we did. the TIGRE toolbox is introduced in the paper "TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction" by Ander Biguri et al. and we refer to this work to produce dataset. The installation link for this toolbox is: https://gitee.com/qiangge_666/TIGRE/blob/master/Frontispiece/MATLAB_installation.md. Both python and matlab installation methods are used, and we used matlab.

After installing the software, you can generate the same data as ours by following these two simple steps. 

First, read and save the CT data in MATLAB format (.mat) by locating the data path under the installation directory at ".\TIGRE-master\MATLAB\Test_data\MRheadbrain\headPhantom.m". Once you have located the data path, load the CT data into your code below the comment line "% load data". 

Second, navigate to the path ".\TIGRE-master\MATLAB\Demos\d03_generateData.m" and set the angle and quantity in the code line "angles=linspace();". For example, setting "angles=linspace(0,2*pi,20);" means generating 20 images in the range of 0 to 360°. Finally, run the code to obtain the projection images with different angles.


Evaluation metrics: 

We propose DisE_algorithm and SAc_alogrithm to evaluate the accuracy of 3D reconstruction, and provide the pseudo-code of these two algorithms.


Data enhancement: 

We propose bone_feature_enhancement_algorithm to enhance data features, and provide python code and pseudo-code, as well as some image examples.


Training details:

1. Read the data. Use the new_try_to_read_image_and_csv.py in SP-net/code folder to read the data.
2. Train the model using train.py in the SP-net/code folder.
3.Use my_evaluation.py under SP-net/code folder to evaluate the model.
The SP-net/code/model folder contains various configurations we propose. To get more information please contact us (1240888547@qq.com). We will add links after the paper is published, so please follow us.
