# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:49:21 2018

@author: lenovo
"""

import numpy as np
import keras.backend
import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Input, Lambda, Dense
import keras.backend as K
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers.pooling import AveragePooling2D, MaxPooling2D


class Mask_loc(keras.engine.topology.Layer):
    def __init__(self, maximum_region, input_shape, **kwargs):
        super(Mask_loc, self).__init__(**kwargs)
        self.W = input_shape[0][0]
        self.H = input_shape[0][1]
        self.q_H = self.H // 4
        self.q_W = self.W // 4
        self.maximum_region = maximum_region

    def call(self, x):
        # image (N,W,H,RGB)
        N = x[0].shape[0]
        print(self.W)
        print(self.H)
        # center coordinate (N,self.maxi_region,2)
        output_region = x[1]
        print(output_region.shape)
        x_cor = output_region[:, :, 0]
        x_cor = tf.clip_by_value(x_cor, self.q_W, 3 * self.q_W)
        y_cor = output_region[:, :, 1]
        y_cor = tf.clip_by_value(y_cor, self.q_H, 3 * self.q_H)
        #       H_tensor = tf.constant(self.q_H,shape = output_region.shape[:2],dtype = tf.float32)
        #       W_tensor = tf.constant(self.q_W,shape = output_region.shape[:2],dtype = tf.float32)
        x_l = tf.tile(tf.expand_dims(x_cor - self.q_W, axis=2),
                      [1, 1, self.W])
        x_r = tf.tile(tf.expand_dims(x_cor + self.q_W, axis=2),
                      [1, 1, self.W])
        y_b = tf.tile(tf.expand_dims(y_cor - self.q_H, axis=2),
                      [1, 1, self.H])
        y_u = tf.tile(tf.expand_dims(y_cor + self.q_H, axis=2),
                      [1, 1, self.H])
        # output = V.concatenate( [tf.expand_dims(t, 2) for t in [x_l,x_r,y_b,y_u]],axis = 2)
        height = tf.reshape(tf.constant(np.arange(self.H).tolist() * self.maximum_region, dtype=tf.float32),
                            (1, self.maximum_region, -1))
        width = tf.reshape(tf.constant(np.arange(self.W).tolist() * self.maximum_region, dtype=tf.float32),
                           (1, self.maximum_region, -1))
        M_x = tf.expand_dims(tf.sigmoid(width - x_l) - tf.sigmoid(width - x_r), axis=3)
        M_y = tf.expand_dims(tf.sigmoid(height - y_u) - tf.sigmoid(height - y_b), axis=3)
        M_x = tf.tile(M_x, [1, 1, 1, self.W])
        M_y = tf.tile(M_y, [1, 1, 1, self.H])
        print(M_x.shape, M_y.shape)
        Mask = tf.expand_dims(tf.multiply(M_x, M_y), axis=4)
        image = tf.expand_dims(x[0], axis=1)

        x = tf.tile(image, [1,self.maximum_region, 1, 1, 1])
        output = tf.multiply(Mask, x)
        print(output.shape, 'output')

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],
                input_shape[1][1],
                input_shape[0][1],
                input_shape[0][2],
                input_shape[0][3])

    def build(self, input_shape):
        super(Mask_loc, self).build(input_shape)


class DAE(keras.models.Model):
    def __init__(self, max_region=4,
                 input_shape=None,
                 ):
        # (3,256,256)
        self.max_region = 4
        self.w = input_shape[0]
        self.h = input_shape[1]
        image = Input(shape=input_shape, name='input')
        print(image.shape, image)
        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }
        bass = VGG16(weights='imagenet', include_top=True)
        base = keras.models.Model(inputs=bass.input,
                                  outputs=[bass.get_layer(index=-5).output, bass.output])
        #####
        bass.trainable = False
        #####
        '''
        x = Conv2D(128,(3,3),stride =2,padding = 'same',activation = 'relu')(image)
        x = MaxPooling2D()(x)
        x = Conv2D(256,(3,3),stride =2,padding = 'same',activation = 'relu')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(512,(3,3),stride =2,padding = 'same',activation = 'relu')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(1024,(3,3),stride =2,padding = 'same',activation = 'relu')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(2048,(3,3),stride =2,padding = 'same',activation = 'relu')(x)
        ouput_features = MaxPooing2D()(x)
        '''

        output_features, _ = base(image)

        print(output_features)

        print(output_features.shape)
        conv_3x3 = Conv2D(
            filters=512,
            name="3x3",
            **options
        )(output_features)
        print(conv_3x3.shape)
        if conv_3x3.shape[2]>1:
            avg_pooling = AveragePooling2D((8, 8))(conv_3x3)
        else:
            avg_pooling = conv_3x3
        output_region = Conv2D(
            filters=max_region * 2,
            kernel_size=(1, 1),
            activation="linear",
            kernel_initializer="zero",
            name="region_proposal"
        )(avg_pooling)
        reshape_layer = Lambda(lambda x: K.reshape(x, (-1, self.max_region, 2)))
        output_region = reshape_layer(output_region)
        print(output_region.shape)
        # To do location_function
        output_proposal_regions = Mask_loc(
            maximum_region=self.max_region,
            input_shape=[input_shape, output_region.shape]
            #   minimum_size=minimum_size
        )([image, output_region])
        # 把locate 到的feature map stack在一起
        print(output_proposal_regions.shape, 'output_prop_region')
        getindicelayer1 = Lambda(lambda x: x[:, 0, :, :, :],
                                 output_shape=lambda input_shape: [input_shape[i] for i in range(0, len(input_shape)) if
                                                                   i != 1])
        part_1 = getindicelayer1(output_proposal_regions)
        print(part_1.shape, part_1)

        p1_encoded, p1 = base(part_1)


        print(p1_encoded, '1 encoded')
        print(p1, 'p1')

        getindicelayer2 = Lambda(lambda x: x[:, 1, :, :, :],
                                 output_shape=lambda input_shape: [input_shape[i] for i in range(0, len(input_shape)) if
                                                                   i != 1])

        part_2 = getindicelayer2(output_proposal_regions)
        print("www: ",type(part_2))
        print(part_2.shape, part_2)
        p2_encoded, p2 = base(part_2)
        print('p2 base pass')

        getindicelayer3 = Lambda(lambda x: x[:, 2, :, :, :],
                                 output_shape=lambda input_shape: [input_shape[i] for i in range(0, len(input_shape)) if
                                                                   i != 1])
        part_3 = getindicelayer3(output_proposal_regions)
        print(part_3.shape, part_3)
        p3_encoded, p3 = base(part_3)
        print('p3 base pass')


        getindicelayer4 = Lambda(lambda x: x[:, 3, :, :, :],
                                 output_shape=lambda input_shape: [input_shape[i] for i in range(0, len(input_shape)) if
                                                                   i != 1])
        part_4 = getindicelayer4(output_proposal_regions)
        print(part_4.shape, part_4)
        p4_encoded, p4 = base(part_4)
        print('p4 base pass')
        mul_output = Lambda(lambda x: K.concatenate([tf.expand_dims(t, 1) for t in x], axis=1))
        score = Lambda(lambda x: K.mean(K.concatenate(
            [tf.expand_dims(t, 1) for t in x], axis=1),
            axis=1))

        outputs_encoded_image = mul_output([p1_encoded,
                                            p2_encoded,
                                            p3_encoded,
                                            p4_encoded])
        outputs_score = score([p1, p2, p3, p4])
        print(outputs_score)
        '''
        def mul_output(x):
            l = []
            for i in range(self.max_region):
                outputs =  self.encoder(x[:,i,:,:,:])
                l.append(outputs)
            return K.concatenate( [tf.expand_dims(t, 1) for t in l],axis = 1)
        def mul_output_shape(input_shape):
            return input_shape
        myCustomLayer = Lambda(mul_output, output_shape=mul_output_shape)
        outputs = myCustomLayer(output_proposal_regions)
        '''
        # print(outputs.shape)
        # assert self.encoder.get_output_at(0) == output_features
        # assert self.encoder.get_output_at(1) == outputs
        # assert self.encoder.get_input_shape_at(0) == (None,256,256,3)
        # assert self.encoder.get_input_shape_at(1) == (None,256,256,3)
        # outputs = K.concatenate( [tf.expand_dims(t, 1) for t in l],axis = 1)
        super(DAE, self).__init__(image, outputs_score)

    def build_encoder(self, top=False):
        def f(x):
            y = ResNet50(weights='imagenet',
                         include_top=False,
                         input_tensor=x).output
            print(y)
            conv_out = Conv2D(1000, (1, 1), activation='relu')(y)
            print(conv_out.shape, 'conv_out')
            return keras.models.Model(inputs=x, outputs=conv_out)

        return f

    def build_vgg_encoder(self):
        def f(x):
            encoder = VGG16(weights='imagenet',
                            include_top=True,
                            input_tensor=x).output
            print(encoder.shape)
            conv_1 = Conv2D(4096, (8, 8), activation='relu')(encoder)
            conv_2 = Conv2D(4096, (1, 1), activation='relu')(conv_1)
            conv_3 = Conv2D(1000, (1, 1), activation='relu')(conv_2)

            return keras.models.Model(inputs=x, outputs=conv_3)

        return f

    def compile(self, optimizer, **kwargs):
        super(DAE, self).compile(optimizer, lossx,None)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        return super(DAE, self).predict(x, batch_size, verbose, steps)


def lossx(y_true,y_pred):
    return K.categorical_crossentropy(y_true,y_pred)


#x = DAE(input_shape = (256,256,3),max_region =4)
#x.summary()