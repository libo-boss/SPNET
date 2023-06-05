# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:30:06 2022

@author: Administrator
"""
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

test_label=np.load(file="D:\\reconstruction\\coronal_normal_1\\dataset\\test_F.npy")

i=0
out = np.load("D:\\reconstruction\\coronal_normal_1\\train_npy\\"+str(i)+".npy", allow_pickle=True) 
test=test_label[i]
con=0
ant=0
epoch=10000

t=0
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
                
# mse_abs=mean_squared_error(out[t],test)
# print('mse_abs:',mse_abs)
# R2_abs=r2_score(out[t],test)########abs_r2  
# print('R2_abs:',R2_abs)            
        
    
    
    