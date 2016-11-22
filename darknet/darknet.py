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
    extension = '.weights'

    def __init__(self, FLAGS):
        self.model = FLAGS.model
        self.get_weight_src(FLAGS)
        
        print 'Parsing {}'.format(self.src_cfg)
        src_parsed = self.parse_cfg(self.src_cfg, FLAGS)
        self.src_meta, self.src_layers = src_parsed
        
        if self.src_cfg == self.model:
            self.meta, self.layers = src_parsed
        else: 
        	print 'Parsing {}'.format(self.model)
        	des_parsed = self.parse_cfg(self.model, FLAGS)
        	self.meta, self.layers = des_parsed

    def get_weight_src(self, FLAGS):
        """
        analyse FLAGS.load to know
        where is the source binary
        and what is its config.
        can be: None, FLAGS.model, or some other
        """
        self.src_bin = self.model + self.extension
        self.src_bin = FLAGS.binary + self.src_bin
        self.src_bin = os.path.abspath(self.src_bin)
        exist = os.path.isfile(self.src_bin)

        if FLAGS.load == str(): FLAGS.load = int()
        if type(FLAGS.load) is int:
            self.src_cfg = self.model
            if FLAGS.load: self.src_bin = None
            elif not exist: self.src_bin = None
        else:
            assert os.path.isfile(FLAGS.load), \
            '{} not found'.format(FLAGS.load)
            self.src_bin = FLAGS.load
            self.src_cfg = loader.model_name(FLAGS.load)
            FLAGS.load = int()


    def parse_cfg(self, model, FLAGS):
        """
        return a list of `layers` objects (darkop.py)
        given path to binaries/ and configs/
        """
        args = [model, FLAGS.binary, FLAGS.config]
        cfg_layers = cfg_yielder(*args)
        meta = dict(); layers = list()
        for i, info in enumerate(cfg_layers):
            if i == 0: meta = info; continue
            else: new = create_darkop(i-1, *info)
            layers.append(new)
        return meta, layers

    def load_weights(self):
        """
        Use `layers` and Loader to load .weights file
        """        
        args = [self.src_bin, self.src_layers]
        wgts_loader = loader.create_loader(*args)

        print ('Loading {} ...'.format(self.src_bin))
        start = time.time()
        for layer in self.layers: layer.load(wgts_loader)
        stop = time.time()
        print ('Finished in {}s'.format(stop - start))