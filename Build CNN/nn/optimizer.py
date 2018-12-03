
"""
This file consists of classes of first order algorithms for neural network optimization.
The base class optimized
"""

import abc
import numpy as np


class Optimizer(object):
    r"""Base class for optimizers
    Args:
        param_groups: a list of all the parameters in the neural network models
        configs: optimization hyper-parameters
    """

    def __init__(self, param_groups):
        self.param_groups = param_groups

    @abc.abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    """Stochastic gradient descent
    """


    def __init__(self, param_groups, lr=5e-3, weight_decay=0.0, momentum=0.4):

        super(SGD, self).__init__(param_groups)
        self.configs = dict(lr=lr, weight_decay=weight_decay,momentum=momentum)

    def step(self):
    # momentum or sgd
        lr = self.configs['lr']
        weight_decay = self.configs['weight_decay']
        momentum = self.configs['momentum']
        # TODO： add momentum term in the algorithm
        # For achieving this goal, you can add more fields in the param groups
        for k,p in self.param_groups.items():
            v = p['v']
            x = p['param']
            g = p['grad']
            g = np.reshape(g,v.shape)
            if weight_decay >0:
                if np.all(v) ==0:
                    v = lr * g
                else:
                    v = momentum * v -lr*g
                    x -= v + weight_decay*x
            else:
                x -= lr*g
            #print(type(x))
            #print(v.shape)
            #print(g.shape)
            #print(type(v))
            self.param_groups[k]['v'] = v
            self.param_groups[k]['param'] = x
        return self.param_groups
