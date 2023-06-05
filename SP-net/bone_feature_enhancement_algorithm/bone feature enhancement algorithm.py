# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:51:03 2022

@author: Administrator
"""


import cv2
import numpy as np
import os
from numpy import *
from numpy import uint8
from pandas import read_csv

import os




    
########图片路径
train_x1='.\\'
train_x2='.\\'
#####读取成矩阵

if not os.path.exists(train_x2):
    os.makedirs(train_x2)
# if not os.path.exists(train_x2+str(k)):
#     os.makedirs(train_x2+str(k))

for i in os.listdir(train_x1):
    # print('i:',i)
    img1 = cv2.imread(train_x1+'\\'+str(i))
    print('img1:',train_x1+'\\'+'\\'+str(i))
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    print('gray1:',gray1.shape)
    mi=np.min(gray1)
    ma=np.max(gray1)
    count = np.bincount(gray1[:,640])  # 找出第3列最频繁出现的值
    fre = np.argmax(count)
    print('fre:',fre)
    me=np.mean(gray1)
    bb=(fre+ma)/2
    cc=(me+ma)/2
    res1=cv2.resize(gray1,(128,1280),interpolation=cv2.INTER_CUBIC)
    train_1 = np.array(gray1)
    # print(train_1.max())
    mad=np.median(gray1)
    if fre>me:
        train_1[train_1 <= fre] = mi
    else:
        train_1[train_1 <= me] = mi
    # train_1[train_1 < 200] = 50
    if bb>cc:
        train_1[train_1 >= cc] = ma
    else:
        train_1[train_1 >= bb] = ma
    cv2.imwrite(train_x2+'\\'+str(i), train_1)
        
    # for i in os.listdir(train_x1+str(k)):
        
    #     img1 = cv2.imread(train_x1+str(k)+'\\'+'\\'+str(i))
    #     print('img1:',train_x1+str(k)+'\\'+str(i))
    #     gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #     res1=cv2.resize(gray1,(768,768),interpolation=cv2.INTER_CUBIC)
    #     train_1 = np.array(gray1)
    #     print(train_1.max())
    #     train_1[train_1 < 100] = 0
    #     # train_1[train_1 < 200] = 50
    #     train_1[train_1 >= 200] = 255
    #     cv2.imwrite(train_x2+str(k)+'\\'+str(i), train_1)
        
# for k in range (35,44):
    
#     ########图片路径
#     train_x1='D:\\reconstruction\\scoliosis\\ct_toatl_jpg\\'
#     train_x2='D:\\reconstruction\\scoliosis\\seg_toatl_jpg\\'
#     #####读取成矩阵
    
#     if not os.path.exists(train_x2+str(k)):
#         os.makedirs(train_x2+str(k))
#     # if not os.path.exists(train_x2+str(k)+'/generator_90'):
#     #     os.makedirs(train_x2+str(k)+'/generator_90')
    
#     for i in os.listdir(train_x1+str(k)):
#         print('i:',i)
#         img1 = cv2.imread(train_x1+str(k)+'\\'+str(i))
#         print('img1:',train_x1+str(k)+'\\'+str(i))
#         gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#         count = np.bincount(gray1[:,256])  # 找出第3列最频繁出现的值
#         fre = np.argmax(count)
#         res1=cv2.resize(gray1,(768,768),interpolation=cv2.INTER_CUBIC)
#         train_1 = np.array(gray1)
#         print(train_1.max())
#         train_1[train_1 <= fre] = 0
#         # train_1[train_1 < 200] = 50
#         me=np.mean(gray1)
#         bb=(255+me)/2
#         cc=(fre+255)/2
#         if bb>cc:
#             train_1[train_1 >= bb] = 255
#         else:
#             train_1[train_1 >= cc] = 255
#         cv2.imwrite(train_x2+'\\'+str(k)+'\\'+str(i), train_1)
        
    # for i in os.listdir(train_x1+str(k)):
        
    #     img1 = cv2.imread(train_x1+str(k)+'\\'+str(i))
    #     print('img1:',train_x1+str(k)+'\\'+str(i))
    #     gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #     res1=cv2.resize(gray1,(768,768),interpolation=cv2.INTER_CUBIC)
    #     train_1 = np.array(gray1)
    #     print(train_1.max())
    #     train_1[train_1 < 100] = 0
    #     # train_1[train_1 < 200] = 50
    #     train_1[train_1 >= 200] = 255
    #     cv2.imwrite(train_x2+str(k)+'\\'+str(i), train_1)