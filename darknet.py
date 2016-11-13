"""
file: darknet.py
includes: definition of class Darknet
this class works with Darknet files: .cfg, .weights
and produces Darknet objects that are easy for TFNet
to use for building the corresponding tensorflow net.

this class uses configs/process.py as a parser for .cfg
files to understand the structure of .weights file. It
will use these information to load all the weights into
its attribute .layers - a well structured list, with each
element is an object of class layer() defined right below
"""

from configs.process import *
import tensorflow as tf
import numpy as np
import time
import os

class layer:
    def __init__(self, type):
        self.type = type
        # any trainable var goes in here:
        self.p = dict()

    def load(self, loader):
        pass

class dropout_layer(layer):
    def __init__(self, p):
        layer.__init__(self, 'dropout')
        self.prob = p

class maxpool_layer(layer):
    def __init__(self, size, stride, pad):
		layer.__init__(self, 'maxpool')
		self.size = size
		self.stride = stride
		self.pad = pad

class Loader(object):
    """
    an iterative reader of .weights files.
    takes path to .weights file `weight_path`,
    reads and returns `length` float32, starting 
    from byte position `offset`
    """
    def __init__(self, weight_path, offset):
        self.offset = offset
        self.weight_path = weight_path
    def __call__(self, length):
        float32_1D_array = np.memmap(self.weight_path, 
            mode= 'r', offset= self.offset, shape= (), 
            dtype = '({})float32,'.format(length))
        self.offset += 4 * length
        return float32_1D_array

class convolu_layer(layer):
    def __init__(self, size, c, n, h, w, 
        stride, pad, batch_norm ): # <- cryin'
        layer.__init__(self, 'convolutional')
        self.size = size
        self.c, self.n = (c, n) 
        self.h, self.w = (h, w)
        self.stride = stride
        self.pad = pad
        self.batch_norm = bool(batch_norm)
# Convolution with bn: conv -> normalize -> scale -> add bias
# Saving conv with bn: bias -> scale -> mean -> var -> kernel            
    def load(self, loader):
        self.p['biases'] = loader(self.n)
        if self.batch_norm:
            self.p['scale'] = loader(self.n)
            self.p['mean'] = loader(self.n)
            self.p['var'] = loader(self.n)
        kernel_size = self.n*self.c*self.size*self.size
        kernel = loader(kernel_size)
        # reshape
        kernel = np.reshape(kernel,
            [self.n, self.c, self.size, self.size])
        kernel = kernel.transpose([2,3,1,0])
        self.p['kernel'] = kernel

class connect_layer(layer):
    def __init__(self, input_size, output_size):
		layer.__init__(self, 'connected')
		self.output_size = output_size
		self.input_size = input_size

    def load(self, loader):
        self.p['biases'] = loader(self.output_size)
        weight_size = self.output_size*self.input_size
        weight_array = loader(weight_size)
        # reshape
        weight_array = np.reshape(weight_array,
            [self.input_size, self.output_size])
        self.p['weights'] = weight_array


class Darknet(object):

    layers = list()
    model = str()
    partial = bool()

    def __init__(self, model, partial = False):        
        self.partial = partial 
        self.model = model
        self.parse(model)
        
        postfix = int('-' in model) * 'binaries/'
        weight_file = postfix + 'yolo-{}.weights'.format(model)
        print ('Loading {} ...'.format(weight_file))
        start = time.time()
        self.loadWeights(weight_file)
        stop = time.time()
        print ('Finished in {}s'.format(stop - start))

    def parse(self, model):
        """
        Use process.py to build `layers`
        """
        cfg = model.split('-')[0]
        print ('Parsing yolo-{}.cfg'.format(cfg))
        layers = cfg_yielder(cfg)
        for i, info in enumerate(layers):
            if i == 0: self.meta = info; continue
            if len(info) == 1: new = layer(type = info[0])
            if info[0] == 'drop': new = dropout_layer(*info[1:])
            if info[0] == 'conv': new = convolu_layer(*info[1:])
            if info[0] == 'pool': new = maxpool_layer(*info[1:])
            if info[0] == 'conn': new = connect_layer(*info[1:])
            self.layers.append(new)

    def loadWeights(self, weight_path):
        """
        Use `layers` and Loader to load .weights file
        """
        file_len = os.path.getsize(weight_path);         
        loader = Loader(weight_path, 16)
        
        for l in self.layers:
            if l.type in ['convolutional', 'connected']:
                l.load(loader)
              
        # Defensive python right here bietch.
        if loader.offset == file_len:
            msg = 'Successfully identified all {} bytes'
            print msg.format(loader.offset)
        else:
            msg = 'Error: expect {} bytes, found {}' 
            exit(msg.format(loader.offset, file_len))