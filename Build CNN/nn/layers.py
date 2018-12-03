"""
This file contains various layers in neural network architecture.
"""

import abc
import numpy as np
from numba import jit


class Layer(object):
    """
    Base class for neural network layer, this class contains abstract methods, hence should not be instantiated.
    Args:
        params: A dictionary of parameters,
                params[k] == v, where k is the name string of the parameter, and v is a dictionary
                containing its properties:
                    v['param']: parameter value
                    v['grad']: gradient
    """

    def __init__(self):
        self.params = dict()

    @abc.abstractmethod
    def forward(self, x):
        r"""Evaluate input features and return output
        :param x: input features
        :return f(x): output features
        """
        pass

    @abc.abstractmethod
    def backward(self, grad_in, x):
        r"""Compute gradient and backpropagate it. The updated gradient should be stored in the field of self.params for
            future reference.
        :param grad_in: gradient from back propagation
        :param x: input features
        :return grad_x: gradient propagated backward to the next layer, namely gradient w.r.t. x
        """
        pass

    # Make the Layer type callable
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Conv(Layer):
    """Convolutional layer
    Args:
        w: convolutional filter:
        b: bias
    """

    def __init__(self, in_channels, out_channels, height, width, stride=2, padding=1, init_scale=1e-2):
        super(Conv, self).__init__()
        # TODO: initial the parameter and value
        #in_channels = 3
        self.params['w'] = {'param':init_scale*np.random.randn(out_channels,in_channels,height,width)}
        self.params['b'] = {'param':np.zeros((1,out_channels))}
        self.padding = padding
        self.stride = stride
    @jit
    def forward(self, x):
        r"""
        :param x: a 4-d tensor, N_samples by n_Channels by Height by Width
        :return: output of convolutional kernel
        """
        # TODO
        out = None
        N,C,H,W = x.shape
        F,_,HH,WW = self.params['w']['param'].shape
        #assert (W + 2 * self.padding - WW) % self.stride == 0, 'width does not work'
        #assert (H + 2 * self.padding - HH) % self.stride == 0, 'height does not work'
        x_padded = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')
        H_new = 1+(H + 2*self.padding - HH) // self.stride
        W_new = 1 + (W + 2*self.padding -WW) //self.stride
        s = self.stride
        out = np.zeros((N,F,H_new,W_new))

        for n in range(N):
            for f in range(F):
                for j in range(H_new):
                    for k in range(W_new):
                        out[n,f,j,k] = np.sum(x_padded[n, :, j*s:HH+j*s, k*s:WW+k*s] * self.params['w']['param'][f]) +self.params['b']['param'][0,f]
        return out
    @jit
    def backward(self, grad_in, x):

        # TODO
        grad_x = None
        F,C,HH,WW = self.params['w']['param'].shape
        N,_,H,W   = x.shape

        #assert (W + 2*self.padding - WW) % self.stride ==0, 'width does not work'
        #assert (H + 2*self.padding - HH) % self.stride ==0, 'height does not work'

        H_new = 1+ int((H +2 * self.padding - HH)/self.stride)
        W_new = 1+ int((W +2 * self.padding - WW)/self.stride)
        grad_x = np.zeros_like(x)
        dw = np.zeros_like(self.params['w']['param'])
        db = np.zeros_like(self.params['b']['param'])
        db = np.sum(grad_in,axis=(0,2,3))
        db = db.T

        s = self.stride
        x_padded = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        dx_padded= np.pad(grad_x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')


        for i in range(N):
            for f in range(F):
                for j in range(H_new):
                    for k in range(W_new):
                        window = x_padded[i, : , j* s:HH+j*s, k*s:WW+k*s]
                        #db[f] += grad_in[i, f, j, k]
                        dw[f] += window * grad_in[i, f, j, k]
                        dx_padded[i,:,j*s:HH+j*s,k*s:WW+k*s] +=  self.params['w']['param'][f] * grad_in[i,f,j,k]

        grad_x = dx_padded[:,:,self.padding:self.padding+H , self.padding:self.padding+W]
        self.params['w']['grad'] = dw
        self.params['b']['grad'] = db
        return grad_x



class Linear(Layer):
    r"""
    Apply linear transform to features

    Args:
        w: n_in by n_out ndarray
        b: 1 by n_out ndarray
    """

    def __init__(self, in_features, out_features, init_scale=1e-2):

        super(Linear, self).__init__()
        self.params['w'] = {'param':np.zeros((in_features,out_features))}
        self.params['b'] = {'param':np.zeros((out_features))}

    def forward(self, x):
        """
        :param x: input features of dimension [n, d1, d2,..., dm]
        :return: output features
        """

        # TODO: write forward propagation
        out = None
        N = x.shape[0]
        x_row = x.reshape(N,-1)
        out = x_row.dot(self.params['w']['param']) + self.params['b']['param']


        return out

    def backward(self, grad_in, x):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        """
        Backward propagation of linear layer
        """
        # TODO: write backward propagation
        grad_x = None
        #print(grad_in.shape)
        #print(self.params['w']['param'].T.shape)
        #print(self.params['w']['param'].T)
        x_row = x.reshape(x.shape[0],-1)  # (N,D)
        grad_x = np.dot(grad_in,self.params['w']['param'].T)  # (N,D)
        #print(x.shape)
        grad_x=np.reshape(grad_x,x.shape)  #(N, d1, ..., d_k)
        self.params['w']['grad']=np.dot(x_row.T,grad_in)
        self.params['b']['grad']=np.sum(grad_in,axis=0,keepdims=True).T


        return grad_x


class Relu(Layer):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, grad_in, x):
        return grad_in * (x > 0)


class MaxPool(Layer):
    """Max pooling
    """

    def __init__(self, kernel_size, stride=2, padding=0):
        super(MaxPool, self).__init__()
        # TODO: initialize pooling layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def forward(self, x):

        # TODO: write forward propagation
        out = None
        HH,WW = self.kernel_size,self.kernel_size
        s = self.stride
        N,C,H,W = x.shape
        H_new = 1+(H-HH) //s
        W_new = 1+(W-WW) //s
        out = np.zeros((N,C,H_new,W_new))
        for i in range(N):
            for j in range(C):
                for k in range(H_new):
                    for l in range(W_new):
                        window = x[i,j,k*s:HH+k*s,l*s:WW+l*s]
                        out[i,j,k,l] = np.max(window)
        return out



    def backward(self, grad_in, x):
        # TODO: write backward propagation
        dx = None
        #############################################################################
        # TODO: Implement the max pooling backward pass                             #
        #############################################################################
        HH, WW = self.kernel_size,self.kernel_size
        s = self.stride
        p = self.padding
        N,C,H,W = x.shape
        H_new = 1+(H-HH)//s
        W_new = 1+(W-WW)//s
        grad_x = np.zeros_like(x)
        x_padded = np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant')
        for i in range(N):
            for j in range(C):
                for k in range(H_new):
                    for l in range(W_new):
                        window = x_padded[i,j,k*s:k*s+HH,l*s:l*s+WW]
                        m = np.max(window)
                        grad_x[i,j,k*s:k*s+HH,l*s:l*s+WW] = (window==m)*grad_in[i,j,k,l]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return grad_x

