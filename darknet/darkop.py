import numpy as np

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

class layer:
    def __init__(self, type, *args):
        self.type = type
        self.w = dict() # weights
        self.h = dict() # placeholders
        self.a = dict() # moving averages
        self.setup(*args)

    def load(self, loader): pass
    def setup(self, *args): pass

class dropout_layer(layer):
    def setup(self, p):
        self.h['pdrop'] = {
            'feed': p, # for training
            'dfault': 1.0, # for testing
            'size': [] # size of this placeholder
        }

class maxpool_layer(layer):
    def setup(self, size, stride, pad):
        self.stride = stride
        self.size = size
        self.pad = pad

class convolu_layer(layer):
    def setup(self, size, c, n, stride, pad, batch_norm):
        self.batch_norm = bool(batch_norm)
        self.c, self.n = (c, n)
        self.stride = stride
        self.size = size
        self.pad = pad
# Convolution with bn: conv -> normalize -> scale -> add bias
# Saving conv with bn: bias -> scale -> mean -> var -> kernel            
    def load(self, loader):
        self.w['biases'] = loader(self.n)
        if self.batch_norm:
            self.w['scale'] = loader(self.n)
            self.w['mean'] = loader(self.n)
            self.w['var'] = loader(self.n)
        kernel_size = self.n*self.c*self.size**2
        kernel = loader(kernel_size)
        # reshape
        kernel = np.reshape(kernel,
            [self.n, self.c, self.size, self.size])
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel

class connect_layer(layer):
    def setup(self, input_size, output_size):
		self.output_size = output_size
		self.input_size = input_size

    def load(self, loader):
        self.w['biases'] = loader(self.output_size)
        weight_size = self.output_size*self.input_size
        weight_array = loader(weight_size)
        # reshape
        weight_array = np.reshape(weight_array,
            [self.input_size, self.output_size])
        self.w['weights'] = weight_array

darkops = {
    'dropout': dropout_layer,
    'connected': connect_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolu_layer
}

def create_darkop(*args):
    op_type = list(args)[0]
    op_class = darkops.get(op_type, layer)
    return op_class(*args)