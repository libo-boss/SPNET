# -*- coding: utf-8 -*-
"""
Created on Sat May  7 10:26:31 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:16:10 2022

@author: Administrator
"""

import numpy as np
import vtk
import math
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import cv2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#%%
readerSTL = vtk.vtkSTLReader()
readerSTL.SetFileName("E:\\360Downloads\\3d_recon\\coronal_scoliosis\\result\\volume0.stl")
readerSTL.Update()
vertebra = readerSTL.GetOutput()
vertebraMapper = vtk.vtkPolyDataMapper() 
vertebraMapper.SetInputData(vertebra)         # maps polygonal data to graphics primitives

vertebraActor = vtk.vtkLODActor() 
vertebraActor.SetMapper(vertebraMapper)
vertebraActor.GetProperty().EdgeVisibilityOn()
vertebraActor.GetProperty().SetLineWidth(0.3)
p = [0, 0, 0]
centrum_out = []
for i in range(vertebra.GetNumberOfPoints()):
    vertebra.GetPoint(i, p)
    # print(p)
    pi=[p[0],p[1],p[2]]
    centrum_out.append(pi)
     
np.savetxt("centrum_out_output.txt",centrum_out, fmt="%.18f", delimiter=" ")
array_centrum_out=np.array(centrum_out)
print('centrum_out_output_center:',array_centrum_out.shape)
centrum_out_x=np.mean(array_centrum_out[:,0])
centrum_out_y=np.mean(array_centrum_out[:,1])
centrum_out_z=np.mean(array_centrum_out[:,2])
center=[centrum_out_x,centrum_out_y,centrum_out_z]
np.savetxt("centrum_out_output_center.txt",center, fmt="%.18f", delimiter=" ")
#%%
readerSTL = vtk.vtkSTLReader()
readerSTL.SetFileName("E:\\360Downloads\\3d_recon\\coronal_scoliosis\\result\\volume_label.stl")
readerSTL.Update()
vertebra = readerSTL.GetOutput()
vertebraMapper = vtk.vtkPolyDataMapper() 
vertebraMapper.SetInputData(vertebra)         # maps polygonal data to graphics primitives

vertebraActor = vtk.vtkLODActor() 
vertebraActor.SetMapper(vertebraMapper)
vertebraActor.GetProperty().EdgeVisibilityOn()
vertebraActor.GetProperty().SetLineWidth(0.3)
p = [0, 0, 0]
centrum = []
for i in range(vertebra.GetNumberOfPoints()):
    vertebra.GetPoint(i, p)
    # print(p)
    pi=[p[0],p[1],p[2]]
    centrum.append(pi)
     
np.savetxt("centrum_test.txt",centrum, fmt="%.18f", delimiter=" ")
array_centrum=np.array(centrum)
print('array_cecentrum_test_centerntrum:',array_centrum.shape)
centrum_x=np.mean(array_centrum[:,0])
centrum_y=np.mean(array_centrum[:,1])
centrum_z=np.mean(array_centrum[:,2])
center_test=[centrum_x,centrum_y,centrum_z]
np.savetxt("centrum_test_center.txt",center_test, fmt="%.18f", delimiter=" ")
#%%sampling to same number
import numpy as np
import os
   
slice=np.random.choice(array_centrum_out.shape[0],array_centrum.shape[0])
# array=np.loadtxt('centrum_out_output_center.txt') 
# print('采样后形状：',array.shape)
array_centrum_out_sample=np.array(array_centrum_out[slice])
np.savetxt('centrum_out_output_center_sample.txt',array_centrum_out_sample,delimiter=" ")
#%%
# a=np.loadtxt('centrum_test.txt')
# b=np.loadtxt('centrum_out_output_center_sample.txt')
print('判断：',array_centrum.shape[0]==array_centrum_out_sample.shape[0])
a=array_centrum
a_1=np.array(center)
a_total=[]
b=array_centrum_out_sample
b_1=np.array(center_test)
b_total=[]

for k in range(array_centrum.shape[0]):
    a1=a[k]
    b1=b[k]
    dis_a=np.sqrt(np.sum((a[k]-a_1)*(a[k]-a_1)))
    a_total.append(dis_a)
    dis_b=np.sqrt(np.sum((b[k]-b_1)*(b[k]-b_1)))
    b_total.append(dis_b)
a_dis=np.array(a_total)
b_dis=np.array(b_total)
a_sum=np.sum(a_dis)
b_sum=np.sum(b_dis)
difference=np.absolute(a_sum-b_sum)
print('差值：',difference)
print('平均差值：',difference/array_centrum.shape[0])
print('差值占比：',2*difference/(a_sum+b_sum))

######常规评价指标没用
# mse_abs=mean_squared_error(array_centrum,array_centrum_out_sample)######abs_mse
# R2_abs=r2_score(array_centrum,array_centrum_out_sample)########abs_r2
# print('mse_abs:',mse_abs,'R2_abs:',R2_abs)
#######绘图
so_a_dis=sorted(a_dis)
epochs_a=range(1,len(so_a_dis)+1)
so_b_dis=sorted(b_dis)
epochs_b=range(1,len(so_b_dis)+1)

import matplotlib.pyplot as plt

plt.plot(epochs_a, so_a_dis, 'b--', label='label-spine')
plt.plot(epochs_b, so_b_dis, 'y', label='generator-spine')
plt.title('Distribution of distances to midpoints coronal_scoliosis3')
plt.legend()

plt.figure()

plt.show()



