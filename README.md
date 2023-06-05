# SPNET

In this study, we developed an end-to-end network called SP-Net to extract features from biplane radiographs and reconstruct the 3D structure of the spine. Multiple configurations of the SP-Net were explored to improve the accuracy of the 3D reconstruction. We defined evaluation metrics to compare with several classical 3D model evaluation methods.

Required environment:
To run our code, the following environment is required: Python 3.6, Keras 2.6.0, and TensorFlow 2.6.2.

Dataset:
Our dataset consists of two categories: normal spine (normal) and scoliosis spine (scoliosis). The dataset is located in the SP-Net/dataset folder and we used eight datasets in our paper. We plan to continue uploading additional data for these two categories.

Dataset production:
To produce the same dataset as we did, you can use CT data that is licensed for use and the TIGRE toolbox, which is introduced in the paper "TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction" by Ander Biguri et al. We refer to this work to produce our dataset. 
The installation link for this toolbox is: https://gitee.com/qiangge_666/TIGRE/blob/master/Frontispiece/MATLAB_installation.md. Both Python and MATLAB installation methods are available, and we used MATLAB for our dataset production.

After installing the software, you can generate the same data as ours by following these two simple steps.

First, read and save the CT data in MATLAB format (.mat) by locating the data path under the installation directory at ".\TIGRE-master\MATLAB\Test_data\MRheadbrain\headPhantom.m". Once you have located the data path, load the CT data into your code below the comment line "% load data".

Second, navigate to the path ".\TIGRE-master\MATLAB\Demos\d03_generateData.m" and set the angle and quantity in the code line "angles=linspace();". For example, setting "angles=linspace(0,2*pi,20);" means generating 20 images in the range of 0 to 360°. Finally, run the code to obtain the projection images with different angles.

Evaluation metrics:
We propose DisE_algorithm and SAc_alogrithm to evaluate the accuracy of 3D reconstruction, and provide the pseudo-code of these two algorithms.

Data enhancement:
We propose bone_feature_enhancement_algorithm to enhance data features, and provide python code and pseudo-code, as well as some image examples.

Training details:

1.Read the data. Use the new_try_to_read_image_and_csv.py in SP-net/code folder to read the data.

2.Train the model using train.py in the SP-net/code folder. 
3.Use my_evaluation.py under SP-net/code folder to evaluate the model. The SP-net/code/model folder contains various configurations we propose. 
To get more information please contact us (1240888547@qq.com). We will add links after the paper is published, so please follow us.
The SP-net/code/model folder contains various configurations we propose. To get more information please contact us (1240888547@qq.com). We will add links after the paper is published, so please follow us.
