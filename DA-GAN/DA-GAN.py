# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:54:16 2018

@author: lenovo
"""
#DA -GAN
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import MaxPooling2D,Dense,Input,Dropout,Reshape, Flatten,Concatenate
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D


class DA_GAN():
    def __init__(self,input_shape,alpha,beta):
        #params
        self.input_shape = input_shape
        self.width = input_shape[0]
        self.height = input_shape[1]
        self.channels = input_shape[2]

        self.gf = 32
        self.df = 64
        optimizer = Adam(0.0002, 0.5)
        self.alpha = alpha
        self.beta = beta
        
        self.D1 = self.build_discriminator()
        self.D2 = self.build_discriminator()
        self.D1.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.D2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        
        self.G = self.build_generator()
        
        DAE_S = Input(shape = self.input_shape)
        DAE_T = Input(shape = self.input_shape)
          
        S_prime = self.G(DAE_S)
        T_prime = self.G(DAE_T)
        
        L_s_gan = self.D1(S_prime)
        L_t_gan = self.D2(T_prime)
        self.D1.trainable = False
        self.D2.trainable = False
        
        self.gan = Model(inputs = [DAE_S, DAE_T], 
                       outputs = [L_s_gan,L_t_gan,
                                  S_prime,T_prime])
        self.gan.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                          self.alpha,self.beta],
                            optimizer=optimizer)
        self.gf = 32
        self.df = 64
        
    def build_discriminator(self):
        def d_layer(layer_input,filters,f_size=4):
            d = Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            return d
        
        img = Input(shape=self.input_shape)
        
        d1 = d_layer(img,self.df)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        
        validity = Conv2D(1,kernel_size=4,strides=1,padding='same')(d4)
    
        return Model(img, validity)
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            #d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            #u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.input_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)
    def train():
        return
    def DAE(self):
        return

