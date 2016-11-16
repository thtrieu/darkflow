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
element is an object of class layer() defined in ./darkop.py
"""

from configs.process import *
from darkop import *
import time
import os

class Darknet(object):

    layers = list()
    model = str()
    partial = bool()

    def __init__(self, model, partial = False):        
        self.partial = partial 
        self.model = model
        self.parse(model) # produce self.layers
        self.checkpoint = False # load from a checkpoint?
        
        weight_file = '{}.weights'.format(model)
        print ('Loading {} ...'.format(weight_file))
        start = time.time()
        self.load_weights(weight_file)
        stop = time.time()
        print ('Finished in {}s'.format(stop - start))

    def parse(self, model):
        """
        Use process.py to build `layers`
        """
        print ('Parsing {}.cfg'.format(model))
        layers = cfg_yielder(model)
        for i, info in enumerate(layers):
            if i == 0: self.meta = info; continue
            else: new = create_darkop(*info)
            self.layers.append(new)

    def load_weights(self, weight_path):
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