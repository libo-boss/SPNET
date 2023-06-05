
from keras.layers.convolutional import Convolution2D

from keras.models import Model

from keras.layers import merge,Flatten,Input,concatenate ,Dense,Dropout
from keras import backend as K
# import try_build_layer_reconsitution
from keras.layers.core import Lambda

import json
import math
import os
import cv2
from PIL import Image
import numpy as np
import keras
from keras import layers

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers.normalization.batch_normalization import BatchNormalizationBase as BatchNormalization
from keras.optimizers import adam_v2 as Adam
#from optimizers import Adam_dlr
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd

import scipy
# from tqdm import tqdm
import tensorflow as tf
import gc
from functools import partial
from sklearn import metrics
from collections import Counter

import itertools
#from bilinear_pooling import bilinear_pooling
#from skconv import SKConv
from keras.models import *

from keras.layers import Reshape, Dense, multiply, Concatenate, Conv2D,Conv3D,Conv2DTranspose,Conv3DTranspose, Add, Activation, Lambda
# from keras_applications.resnet_common import ResNet152,ResNet101,ResNet50V2,ResNet101V2,ResNet152V2,ResNeXt50,ResNeXt101
#from tensorflow.keras import layers
#from keras_squeeze_excite_network import se_inception_resnet_v2
#from densenet import DenseNetImageNet161
#import inception_v4
# from keras_applications.resnet import ResNet152
# from keras_applications import vgg16,inception_resnet_v2
# from keras_efficientnets import EfficientNetB5
# from mixup_generator import MixupGenerator
# #注意力模块
# from self_layers import channel_attention,spatial_attention,cbam_block

def FeatureNetwork():
    inp = Input(shape = (128,128, 1))
    inp1 = Input(shape=(128,128, 1))  # 创建输入1
    inp2 = Input(shape=(128,128, 1))
    
    cnn1 = Conv2D(filters=128, kernel_size=(4, 4), strides=2,padding='same')(inp)
    bn1 = BatchNormalization()(cnn1)
    act1 = Activation('relu')(bn1)
    
    cnn2 = Conv2D(filters=128, kernel_size=(3, 3), strides=1,padding='same')(act1)

    # merge_layers1 = concatenate([cnn1, cnn2])
    
    bn2 = BatchNormalization()(cnn2)   
    act2 = Activation('relu')(bn2)
    
    cnn3 = Conv2D(filters=256, kernel_size=(4, 4), strides=2,padding='valid')(act2)
    bn3 = BatchNormalization()(cnn3)
    act3 = Activation('relu')(bn3)
    
    cnn4 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,padding='same')(act3)
    
    # merge_layers3 = concatenate([cnn3, cnn4])
    
    bn4= BatchNormalization()(cnn4)   
    act4 = Activation('relu')(bn4)
    
    cnn5 = Conv2D(filters=256, kernel_size=(4, 4), strides=2,padding='valid')(act4)
    bn5 = BatchNormalization()(cnn5)
    act5 = Activation('relu')(bn5)
    
    cnn6 = Conv2D(filters=128, kernel_size=(3, 3), strides=1,padding='same')(act5)
    
    # merge_layers5 = concatenate([cnn5, cnn6])
    
    bn6= BatchNormalization()(cnn6)   
    act6 = Activation('relu')(bn6)
    
    model=Model(inputs=inp, outputs=act6)
    return model
network0=FeatureNetwork()
network0.summary()
def FeatureNet(reuse=False):  # add  FeatureNet
    if reuse:
        inp = Input(shape = (128,128, 1), name='F_ImageInput')
        inp1 = Input(shape=(128,128, 1), name='F_ImageInput1')  # 创建输入1
        inp2 = Input(shape=(128,128, 1), name='F_ImageInput2')
        cnn1 = Conv2D(filters=128, kernel_size=(4, 4), strides=2,padding='same')(inp)
        bn1 = BatchNormalization()(cnn1)
        act1 = Activation('relu')(bn1)
        
        cnn2 = Conv2D(filters=128, kernel_size=(3, 3), strides=1,padding='same')(act1)
    
        # merge_layers1 = concatenate([cnn1, cnn2])
        
        bn2 = BatchNormalization()(cnn2)   
        act2 = Activation('relu')(bn2)
        
        cnn3 = Conv2D(filters=256, kernel_size=(4, 4), strides=2,padding='same')(act2)
        bn3 = BatchNormalization()(cnn3)
        act3 = Activation('relu')(bn3)
        
        cnn4 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,padding='same')(act3)
        
        # merge_layers3 = concatenate([cnn3, cnn4])
        
        bn4= BatchNormalization()(cnn4)   
        act4 = Activation('relu')(bn4)
        
        cnn5 = Conv2D(filters=256, kernel_size=(4, 4), strides=2,padding='same')(act4)
        bn5 = BatchNormalization()(cnn5)
        act5 = Activation('relu')(bn5)
        
        cnn6 = Conv2D(filters=128, kernel_size=(3, 3), strides=1,padding='same')(act5)
        
        # merge_layers5 = concatenate([cnn5, cnn6])
        
        bn6= BatchNormalization()(cnn6)   
        act6 = Activation('relu')(bn6)
        
        model=Model(inputs=inp, outputs=act6)
        
        model_1 = model(inp1)  # 孪生网络中的一个特征提取分支
        model_2 = model(inp2)  # 孪生网络中的另一个特征提取分支
        # model_3 = model(inp3) 
        merge_layers = concatenate([model_1, model_2])  # 进行融合，使用的是默认的sum，即简单的相加
        
    else:
        input1 = FeatureNetwork()                     # 孪生网络中的一个特征提取
        input2 = FeatureNetwork()                     # 孪生网络中的另一个特征提取
        # input3 = FeatureNetwork()
        # for layer in input2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
        #     layer.name = layer.name + str("_2")
        inp1 = input1.input
        inp2 = input2.input
        merge_layers = concatenate([input1.output, input2.output])        # 进行融合，使用的是默认的sum，即简单的相加
    ########transformation module
    trans1 = Conv2D(filters=448, kernel_size=(5, 5), strides=2,padding='same',name='trans1')(merge_layers)
    bnt1 = BatchNormalization()(trans1)
    actt1 = Activation('relu')(bnt1)
    print('actt1.shape',actt1.shape)
    view = Reshape((1,28,28,28))( actt1)
    # print('view.shape',view.shape)
    trans2 = Conv3DTranspose(filters=512, kernel_size=(5, 5,5), strides=(2, 2, 2),padding='same',name='trans2')(view)
    bnt2 = BatchNormalization()(trans2)
    actt2 = Activation('relu')(bnt2)
    
    ###Generation network
    g1=Conv3DTranspose(filters=256, kernel_size=(3, 3,3),padding='valid', strides=(1, 1, 1),name='g1')(actt2)
    bng1 = BatchNormalization()(g1)
    actg1 = Activation('relu')(bng1)
   
    g2=Conv3DTranspose(filters=200, kernel_size=(3, 3,3),padding='same', strides=(1, 1, 1),name='g2')(actg1)
    bng2 = BatchNormalization()(g2)
    actg2 = Activation('relu')(bng2)
   
    g3=Conv3DTranspose(filters=150, kernel_size=(3, 3,3),padding='same', strides=(1, 1, 1),name='g3')(actg2)
    bng3 = BatchNormalization()(g3)
    actg3 = Activation('relu')(bng3)
   
    g4=Conv3DTranspose(filters=128, kernel_size=(3, 3,3), strides=(1, 1, 1),name='g4')(actg3)
    bng4 = BatchNormalization()(g4)
    actg4 = Activation('relu')(bng4)
   
    g5=Conv3DTranspose(filters=100, kernel_size=(3, 3,3), strides=(1, 1, 1),name='g5')(actg4)
    bng5 = BatchNormalization()(g5)
    actg5 = Activation('sigmoid')(bng5)
   
    g6=Conv3DTranspose(filters=64, kernel_size=(3, 3,3), strides=(1, 1, 1),name='g6')(actg5)
    bng6 = BatchNormalization()(g6)
    actg6 = Activation('relu')(bng6)
   
    # g7=Conv3DTranspose(filters=128, kernel_size=(3, 3,3),padding='valid', strides=(1, 1, 1),name='g7')(actg6)
    # bng7 = BatchNormalization()(g7)
    # actg7 = Activation('relu')(bng7)
   
    # g8=Conv3DTranspose(filters=100, kernel_size=(3, 3,3),padding='same', strides=(1, 1, 1),name='g8')(actg7)
    # bng8 = BatchNormalization()(g8)
    # actg8 = Activation('sigmoid')(bng8)
    
    # g9=Conv3DTranspose(filters=64, kernel_size=(3, 3,3),padding='same', strides=(1, 1, 1),name='g9')(actg8)
    # bng9 = BatchNormalization()(g9)
    # actg9 = Activation('sigmoid')(bng9)
    
    # g10=Conv3D(filters=64, kernel_size=(1, 1,1), strides=(1, 1, 1),padding='same',name='g10')(actg9)
    # bng10 = BatchNormalization()(g10)
    # actg10 = Activation('sigmoid')(bng10)
    
    # outp = Conv2D(filters=64, kernel_size=(1, 1), strides=1,padding='same',name='outp')(actg10)
    # bnt11 = BatchNormalization()(outp)
    # actt11 = Activation('relu')(bnt11)
    # g9=Conv3D(filters=64, kernel_size=(3, 3,3),padding='same', strides=(1, 1, 1),name='g9')(g8)
    # g10=Conv3D(filters=64, kernel_size=(3, 3,3),padding='same', strides=(1, 1, 1),name='g10')(g9)
    # g11=Conv3D(filters=64, kernel_size=(3, 3,3),padding='same', strides=(1, 1, 1),name='g11')(g10)
    # gg=Reshape((64,64,64,1))( g9)
    # bng8 = BatchNormalization()(g9)
    # actg11 = Activation('sigmoid')(g11)
    # print('actg8.shape:',actg8.shape,'g6.shape:',g6.shape)
    model = Model(inputs=[inp1, inp2], outputs=g6)
    return model
    # return {'inputs': [inp1, inp2], 
    #         'outputs': g11,
    #         'model': model}
network=FeatureNet()

# network.summary()

