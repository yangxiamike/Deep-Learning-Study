# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:54:16 2018

@author: lenovo
"""
# DA -GAN
import keras
import numpy as np
from DAE import DAE
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import MaxPooling2D, Dense, Input, Dropout, Reshape, Flatten, Concatenate
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from data_load.Data_loader import Data_loader
import datetime
import os
from matplotlib.pyplot import plt


# from DataLoader import DataLoader

class DA_GAN():
    def __init__(self, input_shape, alpha, beta, **kwargs):
        # params
        # self.data_loader = DataLoader()
        self.input_shape = input_shape
        self.width = self.input_shape[0]
        self.height = self.input_shape[1]
        self.channels = self.input_shape[2]
        self.dataset_name = 'bird2bird'

        #Load Data Generator
        self.data_load = Data_loader()

        #Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        self.alpha = alpha
        self.beta = beta
        optimizer = Adam(0.0002)

        #Calculate output shape of D (PatchGAN)
        patch = int(self.width / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        #build and compile the dicriminators
        self.D1 = self.build_discriminator()
        self.D2 = self.build_discriminator()
        self.D1.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        self.D2.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])

        self.DAE = DAE(input_shape=(256, 256, 3), max_region=4)


        self.G = self.build_generator((256, 256, 2))

        S = Input(shape=self.input_shape)
        T = Input(shape=self.input_shape)

        DAE_S, DAE_S_score = self.DAE(S)
        lambda_layer = keras.layers.Lambda(lambda x: K.reshape(x, (-1, 256, 256, 2)))
        DAE_T, DAE_T_score = self.DAE(T)

        DAE_S = lambda_layer(DAE_S)
        DAE_T = lambda_layer(DAE_T)
        print(DAE_S.shape)
        # self.G.summary()
        S_prime = self.G(DAE_S)
        T_prime = self.G(DAE_T)

        L_s_gan = self.D1(S_prime)
        L_t_gan = self.D2(T_prime)
        self.D1.trainable = False
        self.D2.trainable = False

        self.gan = Model(inputs=[DAE_S, DAE_T],
                         outputs=[L_s_gan, L_t_gan,
                                  S_prime, T_prime])
        self.gan.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                               'mae', 'mae'],
                         loss_weights=[1, 1,
                                       self.alpha, self.beta],
                         optimizer=optimizer)


    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img = Input(shape=self.input_shape)

        d1 = d_layer(img, self.df)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            # d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            # u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.input_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf * 4)
        u2 = deconv2d(u1, d2, self.gf * 2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def train(self,epochs, batch_size = 1, sample_interval=50):
        '''
        discriminator: self.
        :return:
        '''

        start_time = datetime.datetime.now()
        valid = np.ones((batch_size,)+self.disc_patch)
        fake = np.zeros((batch_size,)+self.disc_patch)


        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_load.load_batch(training_num=1000, batch_size=32, is_training=True)):
                # ----------------------
                #  Train Discriminators
                # ----------------------
                # translate images to opposite domain
                fake_B = self.G.predict(imgs_A)
                fake_A = self.G.predict(imgs_B)

                # Train the discriminators  (original images = Real/translated = Fake)
                dA_loss_real = self.D1.train_on_batch(imgs_A,valid)
                dA_loss_fake = self.D1.train_on_batch(fake_A,fake)
                dA_loss = 0.5 * np.add(dA_loss_real,dA_loss_fake)

                dB_loss_real = self.D2.train_on_batch(imgs_B,valid)
                dB_loss_fake = self.D2.train_on_batch(fake_A,fake)
                dB_loss = 0.5*np.add(dB_loss_fake,dB_loss_real)

                # Total discriminator loss
                d_loss = 0.5 * np.add(dA_loss,dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.gan.train_on_batch([imgs_A,imgs_B],
                                                 [valid,valid,
                                                  imgs_A,imgs_B,
                                                  imgs_A,imgs_B])

                elapased_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f, adv: %05f, recon: %05f, id: %05f] time: %s "\
                                                                % (epoch, epochs,
                                                                   batch_i,self.data_load.train_num,
                                                                   d_loss[0], 100*d_loss[1],
                                                                   g_loss[0],
                                                                   np.mean(g_loss[1:3]),
                                                                   np.mean(g_loss[3:5]),
                                                                   np.mean(g_loss[5:6]),
                                                                   elapsed_time))
                if batch_i % sample_interval == 0:
                    #self.sample_images(epoch,batch_i)
                    print('hhhhhhhhh')
        return 0


    def sample_images(self, epoch, batch_i):
        # os.makedirs('%s' % self.dataset_name, exist_ok= True)
        r, c = 2, 3

        imgs_A = self.data_load.load_data(domain = 'A', batch_size = 1, is_Testing=True)
        imgs_B = self.data_load.load_data(domain = 'B', batch_size = 1, is_Testing=True)

        # Translate images to the other domain
        fake_B = self.G.predict(imgs_A)
        fake_A = self.G.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.G.predict(fake_B)
        reconstr_B = self.G.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A,fake_A,reconstr_A,imgs_B,fake_B,reconstr_B])

        np.save('images/%s/%d_%d.npz' %(self.dataset_name, epoch, batch_i),gen_imgs)

        return 0



