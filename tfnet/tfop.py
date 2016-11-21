"""
file: ./tfops.py
includes: convl, batchnorm, dense, maxpool, etc
functions that takes input `x`, layer `l` of type layer
defined in ./darknet.py and return the output of the
corresponding layer.
"""

from framework import *

def _shape(tensor): # work for both tf.Tensor & np.ndarray
	if type(tensor) is tf.Variable: return tensor.get_shape()
	else: return tensor.shape

class tfop(object):
	"""
	tfop objects initialise with a darknet's `layer` object
	and input tensor of that layer `x`, it calculates the 
	output of this layer and place the result in self.x
	self.x is returned whenever the object is called.
	"""
	def __init__(self, l, x, name, feed = None):
		if feed is not None: self.wrap(l, feed, name)
		if 'tfnetoutput' in name: name = 'output'
		
		self.l = l; self.inp_layer = False
		self.inp_layer = x.name.split(':')[0] == 'input'
		self.inp_size = x.get_shape()
		self.forward(l, x, name)

	def __call__(self, verbalise = True):
		if verbalise: self.verbalise()
		return self.x

	def wrap(self, layer, feed, name):
		"""
		wraps `layer` into tf variables & placeholders
		if layer does not carry value (partial net loaded)
		the corresponding tf variable will also be initialised 
		"""
		for var in layer.w: # trainable vars
			sig = '{}-{}'.format(name, var) # signature
			val = layer.w.get(var, None) # can be a np array or None
			if val is None: # darknet is partially loaded
				args = [layer.shape[var], 0., 1e-2]
				val = tf.truncated_normal(*args)
			layer.w[var] = tf.Variable(val, name = sig)
		
		for ph in layer.h: # placeholders
			sig = '{}-{}'.format(name, ph) # signature/name
			values = layer.h[ph]; shp = layer.shape[ph] # ph shape
			layer.h[ph] = tf.placeholder(tf.float32, shp, sig)
			feed[layer.h[ph]] = values

	def verbalise(self):
		self.detail()
		form = '{:<40} : {}' # verbalise template
		if self.inp_layer: # this is the 1st layer
			print form.format('Input size', self.inp_size)
		print form.format(self.msg, self.x.get_shape())

class conv(tfop):
	def forward(self, l, x, name):
		if l.pad < 0: # figure the pad out
			l.size = l.shape['kernel'][0]
			size = np.int(x.get_shape()[1])
			expect = -(l.pad + 1) * l.stride 
			expect += l.size - size
			padding = [expect / 2, expect - expect / 2]
			if padding[0] < 0: padding[0] = 0
			if padding[1] < 0: padding[1] = 0
		else:
			padding = [l.pad, l.pad]
		x = tf.pad(x, [[0, 0], padding, padding, [0, 0]])
		x = tf.nn.conv2d(x, l.w['kernel'], padding = 'VALID', 
	        name = name, strides = [1, l.stride, l.stride, 1])
		if l.batch_norm: x = self.batchnorm(l, x, name+'-bnorm')
		self.x = tf.nn.bias_add(x, l.w['biases'])
		self.pad = padding

	def batchnorm(self, l, x, name):                
		return tf.nn.batch_normalization(
			x = x, mean = l.w['mean'], variance = l.w['var'], 
			offset = None, scale = l.w['scale'], name = name,
			variance_epsilon = 1e-6)

	def detail(self):
		msg = 'conv{}'.format(_shape(self.l.w['kernel']))
		self.msg = '{:<23} pad{}'.format(msg, self.pad)

class full(tfop):
	def forward(self, l, x, name):
		self.x = tf.nn.xw_plus_b(x, l.w['weights'], 
			l.w['biases'], name = name)

	def detail(self):
		self.msg = 'full{}'.format(_shape(self.l.w['weights']))

class flatten(tfop):
	def forward(self, l, x, name):
		x = tf.transpose(x, [0,3,1,2])
		self.x = slim.flatten(x, scope = name)

	def detail(self):
		self.msg = 'flat()'

class maxpool(tfop):
	def forward(self, l, x, name):
		self.x = tf.nn.max_pool(x, padding = 'VALID',
	        ksize = [1, l.size, l.size, 1], name = name, 
	        strides = [1, l.stride, l.stride, 1])
	
	def verbalise(self): pass

class leaky(tfop):
	def forward(self, l, x, name):
		self.x = tf.maximum(.1*x, x, name = name)

	def verbalise(self): pass

class dropout(tfop):
	def forward(self, l, x, name):
		self.x = tf.nn.dropout(x, l.h['pdrop'], name = name)

	def verbalise(self): pass

op_types = {
	'convolutional': conv,
	'connected': full,
	'maxpool': maxpool,
	'leaky': leaky,
	'dropout': dropout,
	'flatten': flatten
}

def op_create(*args):
	layer_type = list(args)[0].type
	return op_types[layer_type](*args)