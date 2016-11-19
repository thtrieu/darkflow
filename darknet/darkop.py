import numpy as np
import os

class Loader(object):
    """
    an iterative reader of .weights files.
    takes path to .weights file `weight_path`,
    reads and returns `length` float32, starting 
    from byte position `offset`
    """
    def __init__(self, path):
        self.eof = False # end of file
        self.offset = 16 # current pos
        self.path = path # save the path
        if path is not None:
            self.size = os.path.getsize(path)
        else: self.eof = True

    def __call__(self, length):
        if self.eof: return None
        float32_1D_array = np.memmap(self.path, 
            mode= 'r', offset= self.offset, shape= (), 
            dtype = '({})float32,'.format(length))
        self.offset += 4 * length
        if self.offset == self.size: # eof reached 
            self.eof = True
        return float32_1D_array

class layer:
    def __init__(self, type, *args):
        self.sig = [type, args]
        self.type = type
        self.w = dict() # weights
        self.h = dict() # placeholders
        self.a = dict() # moving averages
        self.shape = dict() # weight shape
        self.size = dict() # weight size
        self.setup(*args) # set attr up
        self.cal_size() # calculate sizes

    def load(self, loader): pass
    def setup(self, *args): pass
    def cal_size(self):
        for var in self.shape: \
        self.size[var] = np.prod(self.shape[var])

    # For comparing two layers
    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False
    def __ne__(self, other):
        return not self.__eq__(other)

class maxpool_layer(layer):
    def setup(self, size, stride, pad):
        self.stride = stride
        self.size = size
        self.pad = pad

class dropout_layer(layer):
    def setup(self, p):
        self.h['pdrop'] = {
            'feed': p, # for training
            'dfault': 1.0 # for testing
        }
        self.shape = {
            'pdrop': [] # size of this placeholder
        }

class convolu_layer(layer):
    def setup(self, size, c, n, stride, pad, batch_norm):
        self.batch_norm = bool(batch_norm)
        self.stride = stride
        self.pad = pad
        self.kshape = [n, c, size, size] # darknet shape
        self.shape = {
            'biases': [n],  'var': [n],
            'scale':  [n], 'mean': [n],
            'kernel': [size, size, c, n] # tf shape
        }

# Convolution with bn: conv -> normalize -> scale -> add bias
# Saving conv with bn: bias -> scale -> mean -> var -> kernel            
    def load(self, loader):
        self.w['biases'] = loader(self.size['biases'])
        if self.batch_norm:
            self.w['scale'] = loader(self.size['scale'])
            self.w['mean'] = loader(self.size['mean'])
            self.w['var'] = loader(self.size['var'])
        self.w['kernel'] = loader(self.size['kernel'])
        
        if self.w['kernel'] is None: return
        self.w['kernel'] = self.w['kernel'].reshape(self.kshape)
        self.w['kernel'] = self.w['kernel'].transpose([2,3,1,0])

class connect_layer(layer):
    def setup(self, input_size, output_size):
        self.shape = {
            'biases': [output_size],
            'weights': [input_size, output_size]
        }

    def load(self, loader):
        self.w['biases'] = loader(self.size['biases'])
        self.w['weights'] = loader(self.size['weights'])

        if self.w['weights'] is None: return
        self.w['weights'] = self.w['weights'].reshape(
            self.shape['weights'])

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