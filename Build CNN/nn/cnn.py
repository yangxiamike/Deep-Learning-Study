
from __future__ import division, print_function
import math
import time
from .layers import Conv, Relu, MaxPool, Linear
from .loss import SoftmaxCE,softmax
import numpy as np
import time
from numba import jit


class CNN(object):
    """Convolutional neural network with the following structures:
        conv + relu + pooling + linear + relu + linear + softmax
    """
    def __init__(self, image_size=(3, 32, 32), channels=32, conv_kernel=3,
                 pool_kernel=2, hidden_units=100, n_classes=10):
        """
        :param image_size: an 3 * H * W image, for color image,

        :param channels: channels in the convolution layer
        :param conv_kernel: kernel size of convolutional layer
        :param pool_kernel: kernel size of pooling layer
        :param hidden_units: number of hidden units in linear transform
        """

        # TODO: initialize the neural network. Define the layers
        # Your code should proceed like this:
        #
        # self.conv = Conv(32, 32)
        # self.relu = Relu()
        # ...

        self.h_new = 1+(image_size[1]+2*1-conv_kernel)//2
        self.h_new_new = self.h_new//2
        self.conv = Conv(image_size[0],channels,conv_kernel,conv_kernel)
        self.relu = Relu()
        self.maxpool = MaxPool(pool_kernel)
        self.linear_1 = Linear(channels*self.h_new_new*self.h_new_new,hidden_units)
        #self.linear_1 = Linear(32*7*7,hidden_units)
        self.linear_2 = Linear(hidden_units,n_classes)
        self.softmax = SoftmaxCE()



        # TODO: Add the layers' parameters to the network, which will be assigned to optimizers

        self.param_groups = {
        'w1':{'param':self.conv.params['w']['param'],'v':np.zeros(self.conv.params['w']['param'].shape)}
            ,'b1':{'param':self.conv.params['b']['param'].T,'v':np.zeros(self.conv.params['b']['param'].T.shape)},
        'w2':{'param':self.linear_1.params['w']['param'],'v':np.zeros(self.linear_1.params['w']['param'].shape)}
            ,'b2':{'param':self.linear_1.params['b']['param'].T,'v':np.zeros(self.linear_1.params['b']['param'].T.shape)},
        'w3':{'param':self.linear_2.params['w']['param'],'v':np.zeros(self.linear_2.params['w']['param'].shape)}
            ,'b3':{'param':self.linear_2.params['b']['param'].T,'v':np.zeros(self.linear_2.params['b']['param'].T.shape)}
        }


    def oracle(self, x, y):
        """
        Oracle function to compute value of loss, score and gradient
        :param x: n * c * h * w tensor
        :param y: class label
        :return fx: loss value
        :return s: the output score of each class, this is the output of final linear layer.
        """

        # TODO: Forward propagation
        # In addition to writing the output, you should also receive partial gradient with input of each layer,
        # this will be used in the backpropagation as well.


        c1 = time.clock()
        a1 = self.conv.forward(x)
        c2 = time.clock()
        a2 = self.relu.forward(a1)
        c3 = time.clock()
        a3 = self.maxpool.forward(a2)
        c4 = time.clock()
        a4 = self.linear_1.forward(a3)
        c5 = time.clock()
        a5 = self.relu.forward(a4)
        c6 = time.clock()
        scores = self.linear_2.forward(a5)
        c7 = time.clock()

        data_loss, dscores = self.softmax(scores,y)
        c8 = time.clock()
        da5 = self.linear_2.backward(dscores,a5)
        da4 = self.relu.backward(da5,a4)
        da3 = self.linear_1.backward(da4,a3)
        da2 = self.maxpool.backward(da3,a2)
        da1 = self.relu.backward(da2,a1)
        dx = self.conv.backward(da1,x)

        """
        out = dict()  # a dictionary to receive the partial gradient

        out['w1'] = self.conv.params['w']['grad']
        out['b1'] = self.conv.params['b']['grad']
        out['w2'] = self.linear_1.params['w']['grad']
        out['b2'] = self.linear_1.params['b']['grad']
        out['w3'] = self.linear_2.params['w']['grad']
        out['b3'] = self.linear_2.params['b']['grad']
        """
        self.param_groups['w1']['grad'] = self.conv.params['w']['grad']
        self.param_groups['b1']['grad']=  self.conv.params['b']['grad']
        self.param_groups['w2']['grad'] = self.linear_1.params['w']['grad']
        self.param_groups['b2']['grad'] = self.linear_1.params['b']['grad']
        self.param_groups['w3']['grad'] = self.linear_2.params['w']['grad']
        self.param_groups['b3']['grad'] = self.linear_2.params['b']['grad']


        fx = None  # hold the loss value
        fx = data_loss
        s = None  # hold the score of each class
        scores = softmax(scores)
        s = scores
        time_used = {'conv': c3 - c1,
                     'linear1': c5 - c4,
                     'linear2': c7 - c6,
                     'softmax': c8 - c7,
                     'pool': c4 - c3}

        # TODO: Backward propagation
        return fx,s,time_used

    def score(self, x):
        """
        Score of prediction, a seperate score function is needed in addition to the oracle. It is useful when checking
        accuracy.
        :param x: input features
        :return s: the output score of each class, this is the output of final linear layer.
        """
        # TODO: write the score function

        s = None

        return s