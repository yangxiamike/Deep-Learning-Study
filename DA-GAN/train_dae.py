import numpy as np
import keras
from keras.datasets import cifar100
from DAE import Mask_loc,DAE,lossx
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
from numpy  import tile






def main():
    model = DAE(input_shape = (256,256,3),max_region=4)
    print('1')
    K.set_image_dim_ordering('tf')
    print('2')
    optimizer = Adam(0.0002,0.5)

    model.compile(optimizer=optimizer,loss=lossx,metrics=['accuracy'])
    print('3')
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = tile(x_train,[1,8,8,1])
    x_test = tile(x_test,[1,8,8,1])
    print(x_train.shape)
    y_train = to_categorical(y_train,num_classes=100)
    y_test = to_categorical(y_test,num_classes=100)
    print(y_train)

    model.fit(x=x_train,y=y_train,batch_size=32,epochs=100,validation_data=(x_test,y_test))
    model.save_weights(filepath='/nfsshare/home/xiayang/code/da-gan')


print('1')
main()