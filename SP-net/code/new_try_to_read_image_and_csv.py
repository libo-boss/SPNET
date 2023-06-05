# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 00:17:47 2020

@author: Administrator
"""
import cv2
import numpy as np
import os
from numpy import *
from numpy import uint8
import scipy.io as scio
from scipy.io import loadmat
########图片路径
nn=128
train_x1='D:\\reconstruction\\coronal_normal_4\\dataset\\train\\x1\\'
val_x1='D:\\reconstruction\\coronal_normal_4\\dataset\\validation\\x1\\'
test_x1='D:\\reconstruction\\coronal_normal_4\\dataset\\test\\x1\\'

train_x2='D:\\reconstruction\\coronal_normal_4\\dataset\\train\\x2\\'
val_x2='D:\\reconstruction\\coronal_normal_4\\dataset\\validation\\x2\\'
test_x2='D:\\reconstruction\\coronal_normal_4\\dataset\\test\\x2\\'

train_xF='D:\\reconstruction\\coronal_normal_4\\dataset\\train\\mat\\'
val_xF='D:\\reconstruction\\coronal_normal_4\\dataset\\validation\\mat\\'
test_xF='D:\\reconstruction\\coronal_normal_4\\dataset\\test\\mat\\'
#####读取成矩阵
train_1=[]
for i in os.listdir(train_x1):
    
    data1 = []
    img1 = cv2.imread('D:\\reconstruction\\coronal_normal_4\\dataset\\train\\x1\\'+i)
    res1=cv2.resize(img1,(nn,nn),interpolation=cv2.INTER_CUBIC)
    gray1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
    data1.append(gray1)
    train_1.append(data1)
train_1 = np.array(train_1)
train_1 = np.squeeze(train_1)
#train_1=train_1.reshape(60,656,875,1)
#train_1=train_1.reshape(60,656,875)
print('train_1：',train_1.shape)
#print('训练数据:',train_1)
content_train_1=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\train_1.npy',arr=train_1)
train_2=[]
for i in os.listdir(train_x2):
     
      data2 = []
      img2 = cv2.imread('D:\\reconstruction\\coronal_normal_4\\dataset\\train\\x2\\'+i)
      res2=cv2.resize(img2,(nn,nn),interpolation=cv2.INTER_CUBIC)
      gray2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
      data2.append(gray2)
      train_2.append(data2)
train_2 = np.array(train_2)
train_2 = np.squeeze(train_2)
#train_2=train_2.reshape(60,656,875,1)
print('train_2：',train_2.shape)
content_train_2=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\train_2.npy',arr=train_2)

train_F=[]
for i in os.listdir(train_xF):    
      data3 = []
      tem3='D:\\reconstruction\\coronal_normal_4\\dataset\\train\\mat\\'+i
      data = scio.loadmat(tem3)
      aa=data['instance']
      data3.append(aa)
      train_F.append(data3)
      data3 = np.array(data3)
train_F = np.array(train_F)
train_F = np.squeeze(train_F)
print('train_F：',train_F.shape)
content_train_F=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\train_F.npy',arr=train_F)

val_1=[]
for i in os.listdir(val_x1):    
    data4 = []
    img4 = cv2.imread('D:\\reconstruction\\coronal_normal_4\\dataset\\validation\\x1\\'+i)
    res4=cv2.resize(img4,(nn,nn),interpolation=cv2.INTER_CUBIC)
    gray4 = cv2.cvtColor(res4, cv2.COLOR_BGR2GRAY)
    data4.append(gray4)
    val_1.append(data4)
val_1 = np.array(val_1)
val_1 = np.squeeze(val_1)
#val_1=val_1.reshape(20,656,875,1)
#val_1=val_1.reshape(20,1312,875,1)
print('验证1图像尺寸：',val_1.shape)
content_val_1=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\val_1.npy',arr=val_1)
val_2=[]
for i in os.listdir(val_x2):    
    data5 = []
    img5 = cv2.imread('D:\\reconstruction\\coronal_normal_4\\dataset\\validation\\x2\\'+i)
    res5=cv2.resize(img5,(nn,nn),interpolation=cv2.INTER_CUBIC)
    gray5 = cv2.cvtColor(res5, cv2.COLOR_BGR2GRAY)
    data5.append(gray5)
    val_2.append(data5)
val_2 = np.array(val_2)
val_2 = np.squeeze(val_2)
#val_2=val_2.reshape(20,656,875,1)
#val_2=val_2.reshape(20,1312,875,1)
print('验证2图像尺寸：',val_2.shape)
content_val_2=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\val_2.npy',arr=val_2)
val_F=[]
for i in os.listdir(val_xF):    
      data6 = []
      tem6='D:\\reconstruction\\coronal_normal_4\\dataset\\validation\\mat\\'+i
      data = scio.loadmat(tem6)
      aa=data['instance']
      data6.append(aa)
      val_F.append(data6)
      data6 = np.array(data6)
      
val_F = np.array(val_F)
val_F = np.squeeze(val_F)

#val_F=val_2.reshape(20,1312,875,1)
print('验证标签尺寸：',val_F.shape)
content_val_F=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\val_F.npy',arr=val_F)

test_1=[]
for i in os.listdir(test_x1):    
    data7 = []
    img7 = cv2.imread('D:\\reconstruction\\coronal_normal_4\\dataset\\test\\x1\\'+i)
    res7=cv2.resize(img7,(nn,nn),interpolation=cv2.INTER_CUBIC)
    gray7 = cv2.cvtColor(res7, cv2.COLOR_BGR2GRAY)
    data7.append(gray7)
    test_1.append(data7)
test_1 = np.array(test_1)
test_1 = np.squeeze(test_1)
#test_1=test_1.reshape(20,656*875,1)
#test_1=test_1.reshape(20,1312,875,1)
print('测试1图像尺寸：',test_1.shape)
content_test_1=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\test_1.npy',arr=test_1)
test_2=[]
for i in os.listdir(test_x2):    
    data8 = []
    img8 = cv2.imread('D:\\reconstruction\\coronal_normal_4\\dataset\\test\\x2\\'+i)
    res8=cv2.resize(img8,(nn,nn),interpolation=cv2.INTER_CUBIC)
    gray8 = cv2.cvtColor(res8, cv2.COLOR_BGR2GRAY)
    data8.append(gray8)
    test_2.append(data8)
test_2 = np.array(test_2)
test_2 = np.squeeze(test_2)
#test_2=test_2.reshape(20,656,875,1)
#test_2=test_2.reshape(20,1312,875,1)
print('测试2图像尺寸：',test_2.shape)
content_test_2=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\test_2.npy',arr=test_2)
test_F=[]
for i in os.listdir(test_xF):    
      data9 = []
      tem9='D:\\reconstruction\\coronal_normal_4\\dataset\\test\\mat\\'+i
      data = scio.loadmat(tem9)
      aa=data['instance']
      data9.append(aa)
      test_F.append(data9)
      data9 = np.array(data9)
      
test_F = np.array(test_F)
test_F = np.squeeze(test_F)

#test_F=test_F.reshape(20,1312,875,1)
print('测试标签尺寸：',test_F.shape)
content_test_F=np.save(file='D:\\reconstruction\\coronal_normal_4\\dataset\\test_F.npy',arr=test_F)

print('改变形状后数据:',train_1.shape,val_1.shape,test_1.shape)