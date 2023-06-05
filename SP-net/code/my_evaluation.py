# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:38:53 2022

@author: Administrator
"""
#%%将obj转换为stl
import os
import re 

fileList = []
rot="coronal_normal_3"
file="coronal_normal_3_k"
# 获取特定类型的文件
def getfiles(dirPath, fileType):
    files = os.listdir(dirPath)
    pattern = re.compile('.*\\' + fileType) #设置正则表达式，后缀为fileType
    for f in files:
        # 如果是文件夹，递归调用getfile
        if os.path.isdir(dirPath + '\\' + f):
            getfiles(dirPath + '\\' + f, fileType)
        #  如果是文件，看是否为所需类型
        elif os.path.isfile(dirPath + '\\' + f):
            matches = pattern.match(f) #判断f的文件名是否符合正则表达式，即是否为stl后缀
            if matches != None:
                fileList.append(dirPath + '\\' + matches.group())
        else:
            fileList.append(dirPath + '\\无效文件')
    
    return fileList
test_sample_directory = "D:\\reconstruction\\"+file+"\\result\\"
if not os.path.exists(test_sample_directory):
    os.makedirs(test_sample_directory)

if __name__ == "__main__":
    path = "D:\\reconstruction\\"+file+"\\"
    fType = '.obj'
    res = getfiles(path, fType)
    print('提取结果：')
    os.chdir('E:\\software_install\\mesh_lab\\MeshLab')  # 切换到meshlabserver.exe所在目录
    # aa=0
    # for f in res:
    
    ipath = path+'\\reconstructions\\volume0.obj' # 输入stl模型的路径
    opath = test_sample_directory+ "volume0.stl"  # 输出点云xyz模型的保存路径
    # opath = f[0:-3]+ "xyz"  # 输出点云xyz模型的保存路径
    os.system('meshlabserver -i ' + ipath + ' -o ' + opath + ' -m vn')
    # os.system('meshlabserver -i ' + ipath + ' -o ' + opath + ' -m vn -s　C:\\Users\\admin\\Documents\\lab\\data\\simplify.mlx')
    os.system('exit()') #退出meshlabserver.exe
    # aa=aa+1
#%%计算抽样技术准确率
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

test_label=np.load(file="D:\\reconstruction\\"+rot+"\\dataset\\test_F.npy")

i=0
out = np.load("D:\\reconstruction\\"+file+"\\train_npy\\"+str(i)+".npy", allow_pickle=True) 
t=0
test=test_label[t]
con=0
ant=0
epoch=10000


for j in range (epoch):
    
    a1=np.random.randint(64, size=1)
    a2=np.random.randint(64, size=1)
    a3=np.random.randint(64, size=1)
    # print('选中：',a1,a2,a3)
    # print('检查形状：',out[i].shape,test.shape)
    if test[a1,a2,a3]==out[t][a1,a2,a3]:
        con=con+1
    else:
        ant=ant+1
        # print('异常')
res=con/epoch
print('抽样计数：',res)


count=0
antcou=0
for m1 in range (64):########0-63
    # print('输出：',m1)
    for m2 in range (64):
        for m3 in range (64):
            if test[m1,m2,m3]==out[t][m1,m2,m3]:
                count=count+1
            else:
                antcou=antcou+1
print('完全计数：',count/(64*64*64))
#%%计算欧氏距离
import torch
import numpy as np
import scipy.io as scio
train_1=np.load(file="D:\\reconstruction\\"+file+"\\train_npy\\0.npy")
# train_2=np.load(file="E:\\360Downloads\\3d_recon\\coronal_normal\\dataset\\train_2.npy")
tor=np.array(train_1[0])
print('tor.shape:',tor.shape)
# mm=torch.tensor(tor)
# print('mm.shape:',mm.shape)
dataFile = "D:\\reconstruction\\"+rot+"\\dataset\\test\\mat\\T9_L1_9.mat"
# dataFile1='E:\\360Downloads\\3d_recon\\coronal_normal\\total_mat\\spine\\train\\11.mat'
data = scio.loadmat(dataFile)
aa=data['instance']
print('aa.shape:',aa.shape)
# nn=torch.tensor(aa,dtype=torch.float)
print('aa.shape:',aa.shape)
# m1 = torch.tensor([[1.5, 2.5,3.2], [1.0, 0.5,3.2], [3.2,3.5, 5.5], [3.2,6.5, 7.0]])
# m2 = torch.tensor([[4.5,3.2, 0.5], [9.0,3.2, 10.0], [3.2,4.0, 6.0], [2.5,3.2, 3.2]])
# dist_matrix = torch.cdist(mm, nn)
# print(dist_matrix)
# print('max(dist_matrix):',max(dist_matrix))

def calEuclidean(x, y):
    dist = np.sqrt(np.sum(np.square(x-y)))   # 注意：np.array 类型的数据可以直接进行向量、矩阵加减运算。np.square 是对每个元素求平均~~~~
    return dist

print('欧氏距离：',calEuclidean(tor, aa))
#%%计算平均差值并绘图
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
readerSTL.SetFileName("D:\\reconstruction\\"+file+"\\result\\volume0.stl")
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
readerSTL.SetFileName("D:\\reconstruction\\"+rot+"\\result\\volume0_label.stl")
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
plt.title('Distribution of distances to midpoints coronal_normal_4')
plt.legend()

plt.figure()

plt.show()
#%%计算HD,ASD,SO,VD
import surface_distance as surfdist
import numpy as np
import scipy.spatial.distance as dist
i=0
t=0
test_label=np.load(file="D:\\reconstruction\\"+rot+"\\dataset\\test_F.npy")

out = np.load("D:\\reconstruction\\"+file+"\\train_npy\\"+str(i)+".npy", allow_pickle=True) 

test_label_bool=[test_label > 0.5] 

out_bool=[out > 0.5]

import surface_distance as surfdist
mask_gt=test_label_bool[0][0]
mask_pred=out_bool[0][0]
surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
print('avg_surf_dist:',avg_surf_dist)

import surface_distance as surfdist

surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 99)
print('hd_dist_95:',hd_dist_95)

import surface_distance as surfdist

surface_distances = surfdist.compute_surface_distances(
    mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
print('surface_overlap:',surface_overlap)

import surface_distance as surfdist

surface_distances = surfdist.compute_surface_distances(
    mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)
print('surface_dice:',surface_dice)

import surface_distance as surfdist

volume_dice = surfdist.compute_dice_coefficient(mask_gt, mask_pred)