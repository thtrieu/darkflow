from layer import Layer
from convolution import convolutional_layer, local_layer
from connected import modify_layer, connected_layer

class avgpool_layer(Layer):
    pass

class crop_layer(Layer):
    pass

class maxpool_layer(Layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

class softmax_layer(Layer):
    def setup(self, groups):
        self.groups = groups

class dropout_layer(Layer):
    def setup(self, p):
        self.h['pdrop'] = dict({
            'feed': p, # for training
            'dfault': 1.0, # for testing
            'shape': ()
        })

"""
Darkop Factory
"""

darkops = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolutional_layer,
    'avgpool': avgpool_layer,
    'softmax': softmax_layer,
    'crop': crop_layer,
    'local': local_layer,
    'modify': modify_layer
}

def create_darkop(num, ltype, *args):
    op_class = darkops.get(ltype, Layer)
    return op_class(num, ltype, *args)