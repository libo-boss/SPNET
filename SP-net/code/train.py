#import argparse
#from keras.preprocessing.image import ImageDataGenerator
#import pandas as pd
from generator import data_generator
from model import FeatureNetwork,FeatureNet
# from model import regress
#import read_m1_and_m2
#import new_try_to_read_image_and_csv
#####debug
#from keras.models import load_model
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping,LearningRateScheduler
import os
#from self_layers import ABS_norm,FBN_norm
# from model import regress
#from keras.models import load_model
from utils import npytar,save_volume
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras import backend as K
from keras.activations import sigmoid
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
def weighted_binary_crossentropy(target, output):
    print(target.dtype)
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss


train_1=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\train_1.npy")
train_2=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\train_2.npy")
val_1=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\val_1.npy")
val_2=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\val_2.npy")
test_1=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\test_1.npy")
test_2=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\test_2.npy")
# train_2=new_try_to_read_image_and_csv.train_2
# val_1=new_try_to_read_image_and_csv.val_1
# val_2=new_try_to_read_image_and_csv.val_2
# test_1=new_try_to_read_image_and_csv.test_1
# test_2=new_try_to_read_image_and_csv.test_2
# train_label=np.load(file="E:\\3d_reconstruction_data\\label_train_F.npy")
# val_label=np.load(file="E:\\3d_reconstruction_data\\label_val_F.npy")
# test_label=np.load(file="E:\\3d_reconstruction_data\\label_val_F.npy")
train_label=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\train_F.npy")############用ransac计算的label
val_label=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\val_F.npy")
test_label=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\test_F.npy")

tr_F=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\tr_F.npy")############用ransac计算的label
va_F=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\v_F.npy")
te_F=np.load(file="D:\\reconstruction\\coronal_normal_3_k\\dataset\\te_F.npy")

train_data=data_generator(train_1,train_2,tr_F,train_label,1)
val_data=data_generator(val_1,val_2,va_F,val_label,1)
test_data=data_generator(test_1,test_2,te_F,test_label,1)

model = FeatureNet()
model.summary()
# inputs = model['inputs']
# outputs = model['outputs']
# model = model['model']
# voxel_loss = K.cast(K.mean(weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32') #+ kl_div
# model.add_loss(voxel_loss)

model.compile(loss='mae', optimizer='Adam', metrics=['mae'])
# model.compile( optimizer='adam', metrics=['mae'])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.7,patience=2, min_lr=1e-8, verbose=1)
history =model.fit_generator(train_data,validation_data=val_data,steps_per_epoch=800,validation_steps=10
                              ,epochs=100                            
                              # ,callbacks=[reduce_lr]
                               ,callbacks=[early_stopping]
                              )
# # ######train_data_save

# model.save('E:\\3d_reconstruction_data\\F_ransac_self_2_2weights.hdf5')
model.save_weights('./weigths.h5')
# model.load_weights('./weigths.h5')
test_sample_directory = '.\\train_npy\\'
if not os.path.exists(test_sample_directory):
    os.makedirs(test_sample_directory)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))

# import matplotlib.pyplot as plt

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()
# model1=load_model('E:\\3d_reconstruction_data\\F_ransac_weights.hdf5')


# # model = load_model('E:\\3d_reconstruction_data\\cats_and_dogs_small.hdf5')

# pred=[]
# mse_1=[]
# mse_2=[]
# mae_1=[]
# mae_2=[]
# rmse_1=[]
# rmse_2=[]
# R2_1=[]
# R2_2=[]
if not os.path.exists('image_2'):
    os.makedirs('image_2')
if not os.path.exists('image_1'):
    os.makedirs('image_1')
if not os.path.exists('reconstructions'):
    os.makedirs('reconstructions')
    
if not os.path.exists('reconstructions4'):
    os.makedirs('reconstructions4')
nn=128
if not os.path.exists('reconstructions8'):
    os.makedirs('reconstructions8')
    
    
if not os.path.exists('reconstructions_test1'):
    os.makedirs('reconstructions_test1')
if not os.path.exists('reconstructions_test4'):
    os.makedirs('reconstructions_test4')
if not os.path.exists('reconstructions_test8'):
    os.makedirs('reconstructions_test8')
    


import cv2  
import imageio 
for i in range (0,30):
    print('预测第',i,'个')
    t1=test_1[i].reshape(1,nn,nn,1)
    t2=test_2[i].reshape(1,nn,nn,1)
    t3=te_F[i].reshape(1,9)
    # t3=np.squeeze(t3)
    print('t3.shape:',t3.shape)
    ppp=model.predict([t1,t2,t3])
    ppp = np.squeeze(ppp)
    print('ppp.shape:',ppp.shape)

    print('ppp[1].shape:',ppp[1].shape)
    print('ppp[4].shape:',ppp[4].shape)
    print('ppp[8].shape:',ppp[8].shape)
    # pp = np.squeeze(pp)
    # print('squeeze pp.shape:',pp[0].shape)
    ppp[ppp >= 0.5] = 1
    ppp[ppp < 0.5] = 0
    content_train_1=np.save(file=test_sample_directory+'/'+str(i)+'.npy',arr=ppp)
    save_volume.save_output(ppp[0, :], 64, 'reconstructions', i)
    ####存一个就可以
    # save_volume.save_output(ppp[4, :], 64, 'reconstructions4', i)
    # save_volume.save_output(ppp[8, :], 64, 'reconstructions8', i)
    # pp[i, 0, :].dump(test_sample_directory+'/'+str(i)+'.npy')
#     mse_abs=mean_squared_error(test_label[i],pp[0])######abs_mse
#     mse_fbn=mean_squared_error(test_label[i],pp[1])#######fbn_mse

# for i in range (50,64):
#     print('预测第',i,'个')
#     t1=test_1[i].reshape(1,nn,nn,1)
#     t2=test_2[i].reshape(1,nn,nn,1)
#     ppp=model.predict([t1,t2])
#     ppp = np.squeeze(ppp)
#     print('ppp.shape:',ppp.shape)

#     print('ppp[1].shape:',ppp[1].shape)
#     print('ppp[4].shape:',ppp[4].shape)
#     print('ppp[8].shape:',ppp[8].shape)
#     # pp = np.squeeze(pp)
#     # print('squeeze pp.shape:',pp[0].shape)
#     ppp[ppp > 0.5] = 1
#     ppp[ppp < 0.5] = 0
#     content_train_1=np.save(file=test_sample_directory+'/'+str(i)+'.npy',arr=ppp)
#     save_volume.save_output(ppp[0, :], 64, 'reconstructions', i)
# #####预测第一个
# te1=cv2.imread('D:\\reconstruction\\n11.png')
# te1=cv2.resize(te1,(128,128),interpolation=cv2.INTER_CUBIC)
# te1 = cv2.cvtColor(te1, cv2.COLOR_BGR2GRAY)
# te1=te1.reshape(1,nn,nn,1)
# te2=cv2.imread('D:\\reconstruction\\n12.png')
# te2=cv2.resize(te2,(128,128),interpolation=cv2.INTER_CUBIC)
# te2 = cv2.cvtColor(te2, cv2.COLOR_BGR2GRAY)
# te2=te2.reshape(1,nn,nn,1)
# pre=model.predict([te1,te2])
# ppp = np.squeeze(pre)
# print('测试ppp.shape:',ppp.shape)

# print('ppp[1].shape:',ppp[1].shape)
# print('ppp[4].shape:',ppp[4].shape)
# print('ppp[8].shape:',ppp[8].shape)
# # pp = np.squeeze(pp)
# # print('squeeze pp.shape:',pp[0].shape)
# ppp[ppp > 0.5] = 1
# ppp[ppp < 0.5] = 0
# # content_train_1=np.save(file=test_sample_directory+'/'+str(i)+'.npy',arr=ppp)
# save_volume.save_output(ppp[0, :], 64, 'reconstructions_test1', 0)
# save_volume.save_output(ppp[1, :], 64, 'reconstructions_test4', 1)
# save_volume.save_output(ppp[2, :], 64, 'reconstructions_test8', 2)
# #####预测第二个
# te1=cv2.imread('D:\\reconstruction\\n21.png')
# te1=cv2.resize(te1,(128,128),interpolation=cv2.INTER_CUBIC)
# te1 = cv2.cvtColor(te1, cv2.COLOR_BGR2GRAY)
# te1=te1.reshape(1,nn,nn,1)
# te2=cv2.imread('D:\\reconstruction\\n22.png')
# te2=cv2.resize(te2,(128,128),interpolation=cv2.INTER_CUBIC)
# te2 = cv2.cvtColor(te2, cv2.COLOR_BGR2GRAY)
# te2=te2.reshape(1,nn,nn,1)
# pre=model.predict([te1,te2])
# ppp = np.squeeze(pre)
# print('测试ppp.shape:',ppp.shape)

# print('ppp[1].shape:',ppp[1].shape)
# print('ppp[4].shape:',ppp[4].shape)
# print('ppp[8].shape:',ppp[8].shape)
# # pp = np.squeeze(pp)
# # print('squeeze pp.shape:',pp[0].shape)
# ppp[ppp > 0.5] = 1
# ppp[ppp < 0.5] = 0
# # content_train_1=np.save(file=test_sample_directory+'/'+str(i)+'.npy',arr=ppp)
# save_volume.save_output(ppp[0, :], 64, 'reconstructions_test1', 00)
# save_volume.save_output(ppp[1, :], 64, 'reconstructions_test4', 11)
# save_volume.save_output(ppp[2, :], 64, 'reconstructions_test8', 22)
# #####预测第三个
# te1=cv2.imread('D:\\reconstruction\\n31.png')
# te1=cv2.resize(te1,(128,128),interpolation=cv2.INTER_CUBIC)
# te1 = cv2.cvtColor(te1, cv2.COLOR_BGR2GRAY)
# te1=te1.reshape(1,nn,nn,1)
# te2=cv2.imread('D:\\reconstruction\\n32.png')
# te2=cv2.resize(te2,(128,128),interpolation=cv2.INTER_CUBIC)
# te2 = cv2.cvtColor(te2, cv2.COLOR_BGR2GRAY)
# te2=te2.reshape(1,nn,nn,1)
# pre=model.predict([te1,te2])
# ppp = np.squeeze(pre)
# print('测试ppp.shape:',ppp.shape)

# print('ppp[1].shape:',ppp[1].shape)
# print('ppp[4].shape:',ppp[4].shape)
# print('ppp[8].shape:',ppp[8].shape)
# # pp = np.squeeze(pp)
# # print('squeeze pp.shape:',pp[0].shape)
# ppp[ppp > 0.5] = 1
# ppp[ppp < 0.5] = 0
# # content_train_1=np.save(file=test_sample_directory+'/'+str(i)+'.npy',arr=ppp)
# save_volume.save_output(ppp[0, :], 64, 'reconstructions_test1', 000)
# save_volume.save_output(ppp[1, :], 64, 'reconstructions_test4', 111)
# save_volume.save_output(ppp[2, :], 64, 'reconstructions_test8', 222)
# # save_volume.save_output(ppp[3, :], 64, 'reconstructions_test1', 3)
# # save_volume.save_output(ppp[4, :], 64, 'reconstructions_test4', 4)
# # save_volume.save_output(ppp[5, :], 64, 'reconstructions_test8', 5)
# # save_volume.save_output(ppp[6, :], 64, 'reconstructions_test1', 6)
# # save_volume.save_output(ppp[7, :], 64, 'reconstructions_test4', 7)
# # save_volume.save_output(ppp[8, :], 64, 'reconstructions_test8', 8)
# # save_volume.save_output(ppp[9, :], 64, 'reconstructions_test1', 9)

# # ######运行完就关机
# # from datetime import *
# # import os
# # import time
# # # tmNow = datetime.now()
# # # d = date.today()
# # # t = time(23,10,0)
# # # shtdownTime = datetime.combine(d,t)
# # # print(time.localtime(time.time()))
# # def ShutDown():
# #     while True:
# #         tmNow = datetime.now()
# #         timedDelta = (tmNow - tmNow).total_seconds()
# #         if timedDelta < 60:
# #             os.system('shutdown -s -f -t 59')
# #             break
# #         else:
# #             continue
            

# # ShutDown()



