from utils import loader
import numpy as np

class layer(object):
    def __init__(self, *args):
        self.signature = list(args)
        self.number = list(args)[0]
        self.type = list(args)[1]
        self.w = dict() # weights
        self.h = dict() # placeholders
        self.shape = dict() # weight shape
        self.size = dict() # weight size
        self.setup(*args[2:]) # set attr up
        self.cal_size() # calculate sizes

    def cal_size(self):
        for var in self.shape:
            shape = self.shape[var]
            self.size[var] = np.prod(shape)

    def load(self, src_loader):
        if self.type not in src_loader.VAR_LAYER: return
        if type(src_loader) is loader.weights_loader:
            loaded_layer = src_loader(self)
            sig = self.signature[1:]
            if loaded_layer is not None:
                print 'Re-collect {}'.format(sig)
                self.w = loaded_layer.w
            else: print 'Initialize {}'.format(sig)
            return

        for var in self.shape:
            name = str(self.number)
            name += '-' + self.type
            name += '-' + var
            shape = self.shape[var]
            val = src_loader(name, shape)
            self.w[var] = val
            print 'Re-collect {}: {}'.format(name, shape)

    # For comparing two layers
    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False
    def __ne__(self, other):
        return not self.__eq__(other)

    # Over-ride methods
    def setup(self, *args): pass
    def finalize(self): pass 

class maxpool_layer(layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

class dropout_layer(layer):
    def setup(self, p):
        self.h['pdrop'] = {
            'feed': p, # for training
            'dfault': 1.0, # for testing
            'shape': []
        }

class convolu_layer(layer):
    def setup(self, ksize, c, n, stride, pad, batch_norm):
        self.batch_norm = bool(batch_norm)
        self.stride = stride
        self.pad = pad
        self.kshape = [n, c, ksize, ksize] # darknet shape
        self.shape = {
            'biases': [n], 
            'kernel': [ksize, ksize, c, n]
        }
        if self.batch_norm:
            self.shape.update({
                'var'  : [n], 
                'scale': [n], 
                'mean' : [n]
            })

    def finalize(self):
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.kshape)
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel

class connect_layer(layer):
    def setup(self, input_size, output_size):
        self.shape = {
            'biases': [output_size],
            'weights': [input_size, output_size]
        }

    def finalize(self):
        weights = self.w['weights']
        if weights is None: return
        weights = weights.reshape(
            self.shape['weights'])
        self.w['weights'] = weights

darkops = {
    'dropout': dropout_layer,
    'connected': connect_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolu_layer
}

def create_darkop(*args):
    op_type = list(args)[1]
    op_class = darkops.get(op_type, layer)
    return op_class(*args)