
# coding: utf-8

# In[14]:


# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:54:16 2018

@author: lenovo
"""
#DA -GAN
import keras
import numpy as np
from DAE import DAE
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import MaxPooling2D,Dense,Input,Dropout,Reshape, Flatten,Concatenate
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
#from DataLoader import DataLoader

class DA_GAN():
    def __init__(self,input_shape,alpha,beta,**kwargs):
        #params
        #self.data_loader = DataLoader()
        self.input_shape = input_shape
        self.width = self.input_shape[0]
        self.height = self.input_shape[1]
        self.channels = self.input_shape[2]

        self.gf = 32
        self.df = 64
        
        self.alpha = alpha
        self.beta = beta
        optimizer = Adam(0.0002)
        
        patch = int(self.width / 2**4)
        self.disc_patch = (patch, patch, 1)

        
        self.D1 = self.build_discriminator()
        self.D2 = self.build_discriminator()
        self.D1.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.D2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        self.DAE = DAE(input_shape = (256,256,3),max_region = 4)
        
        self.G = self.build_generator((256,256,2))
        
        S = Input(shape = self.input_shape)
        T = Input(shape = self.input_shape)
        
        DAE_S,DAE_S_score = self.DAE(S)
        lambda_layer = keras.layers.Lambda(lambda x: K.reshape(x,(-1,256,256,2)))
        DAE_T,DAE_T_score = self.DAE(T)
        
        DAE_S = lambda_layer(DAE_S)
        DAE_T = lambda_layer(DAE_T)
        print(DAE_S.shape)
        #self.G.summary()
        S_prime = self.G(DAE_S)
        T_prime = self.G(DAE_T)
        
        L_s_gan = self.D1(S_prime)
        L_t_gan = self.D2(T_prime)
        
        DAE_S_prime,DAE_S_prime_score = self.DAE(S_prime)
        DAE_T_prime,DAE_T_prime_score = self.DAE(T_prime)
        
        self.D1.trainable = False
        self.D2.trainable = False
        print(DAE_S_prime.shape)
        print(DAE_T_prime.shape)
        self.gan = Model(inputs = [S, T], 
                           outputs = [L_s_gan,
                                  L_t_gan,
                                  DAE_S_prime,
                                  DAE_T_prime
                                 ])
        self.gan.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                          self.alpha,self.beta],
                            optimizer=optimizer)
        
        
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
    def build_generator(self,input_shape):
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
        d0 = Input(shape=input_shape)
        print(d0.shape,'d0')
        # Downsampling
        d1 = conv2d(d0, self.gf)
        print(d1.shape,'d1')
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        print(u1.shape,'u1')
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)
    
    
    def train(self,epochs,batch_size):
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        start_time = datetime.datetime.now()
        
        
        for epoch in range(epochs):
            for batch_i, (imgs_S, imgs_T) in enumerate(self.data_loader.load_batch(batch_size)):
                DAE_S = self.DAE(imgs_S)
                DAE_T = self.DAE(imgs_T)
                
                S_prime = self.G(DAE_S)
                T_prime = self.G(DAE_T)
                
                dS_loss_real = self.d1.train_on_batch(imgs_S, valid)
                dS_loss_fake = self.d1.train_on_batch(S_prime, fake)
                dS_loss = 0.5 * np.add(dS_loss_real, dS_loss_fake)

                dT_loss_real = self.d2.train_on_batch(imgs_T, valid)
                dT_loss_fake = self.d2.train_on_batch(T_prime, fake)
                dT_loss = 0.5 * np.add(dT_loss_real, dT_loss_fake)
                d_loss = 0.5 * np.add(dS_loss, dT_loss)
                
                
                g_loss = self.gan.train_on_batch([DAE_S,DAE_T],
                                                 [valid,valid,
                                                  DAE_S,DAE_T])
        
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "                                                                         % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6])))



# In[15]:


K.clear_session()
X = DA_GAN((256,256,3),0.1,0.5)


# In[ ]:


class mae():
    def __init__(self,input_shape,gf):
        self.input_shape = input_shape
        self.gf = gf
        self.channels = self.input_shape[2]
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
        output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)


# In[13]:


y = mae(input_shape = (256,256,20),gf = 16)
f = y.build_generator()
f.summary()

