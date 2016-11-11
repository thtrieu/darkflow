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
    def __init__(self, type, size = 0, 
    	c = 0, n = 0, h = 0, w = 0):
        self.type = type
        self.size = size
        self.c, self.n = (c, n) 
        self.h, self.w = (h, w)

class dropout_layer(layer):
    def __init__(self, p):
        self.type = 'dropout'
        self.prob = p

class btchnrm_layer(layer):
    def __init__(self, size, c, n, h, w ): # <- cryin' haha
        layer.__init__(self, 'batchnorm',
            size, c, n, h, w)

class maxpool_layer(layer):
    def __init__(self, size, c, n, h, w, 
        stride, pad ):
		layer.__init__(self, 'maxpool', 
			size, c, n, h, w)
		self.stride = stride
		self.pad = pad

class convolu_layer(layer):
    def __init__(self, size, c, n, h, w, 
        stride, pad ):
        layer.__init__(self, 'convolutional', 
        	size, c, n, h, w)
        self.stride = stride
        self.pad = pad

class connect_layer(layer):
    def __init__(self, size, c, n, h, w, 
    	input_size, output_size):
		layer.__init__(self, 'connected', 
			size, c, n, h, w)
		self.output_size = output_size
		self.input_size = input_size

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
        cfg = model.split('-')[0]
        print ('Parsing yolo-{}.cfg'.format(cfg))
        layers = cfg_yielder(cfg)
        for i, info in enumerate(layers):
            if i == 0: self.meta = info; continue
            if len(info) == 1: new = layer(type = info[0])
            if info[0] == 'bnrm': new = btchnrm_layer(*info[1:])
            if info[0] == 'drop': new = dropout_layer(*info[1:])
            if info[0] == 'conv': new = convolu_layer(*info[1:])
            if info[0] == 'pool': new = maxpool_layer(*info[1:])
            if info[0] == 'conn': new = connect_layer(*info[1:])
            self.layers.append(new)

    def loadWeights(self, weight_path):
        file_len = os.path.getsize(weight_path); offset = 16

        # Read byte arrays from file
        for i in range(len(self.layers)):
            l = self.layers[i]
            if l.type == "convolutional":
                weight_number = l.n * l.c * l.size * l.size
                l.biases = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(l.n))
                offset += 4 * l.n
                l.weights = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(weight_number))
                offset += 4 * weight_number
            
            elif l.type == "batchnorm":
                l.biases = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(l.n))
                offset += 4 * l.n
                l.weights = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(l.n))
                offset += 4 * l.n

            elif l.type == "connected":
                bias_number = l.output_size
                weight_number = l.output_size * l.input_size
                l.biases = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(bias_number))
                offset += bias_number * 4
                l.weights = np.memmap(weight_path, mode = 'r',
                    offset = offset, shape = (),
                    dtype = '({})float32,'.format(weight_number))
                offset += weight_number * 4
              
        # Defensive python right here bietch.
        if offset == file_len:
            print 'Successfully identified all {} bytes'.format(offset)
        else:
            exit('Error: expect {} bytes, found {}'.format(offset, file_len))

        # Reshape
        for i in range(len(self.layers)):
            l = self.layers[i]
            
            if l.type == 'convolutional':
                weight_array = l.weights
                weight_array = np.reshape(weight_array,
                	[l.n, l.c, l.size, l.size])
                weight_array = weight_array.transpose([2,3,1,0])
                l.weights = weight_array

            if l.type == 'connected':
                weight_array = l.weights
                weight_array = np.reshape(weight_array,
                	[l.input_size, l.output_size])
                l.weights = weight_array