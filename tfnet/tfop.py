"""
file: ./tfops.py
includes: convl, batchnorm, dense, maxpool, etc
functions that takes input `inp`, layer `layer` of type layer
defined in ./darknet/darkop.py and return the output of the
corresponding layer in .out
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def _shape(tensor): # work for both tf.Tensor & np.ndarray
	if type(tensor) in [tf.Variable, tf.Tensor]: 
		return tensor.get_shape()
	else: return tensor.shape

class tfop(object):
	"""
	tfop objects initialise with a darknet's `layer` object
	and input tensor of that layer `x`, it calculates the 
	output of this layer and place the result in self.x
	"""

	def __init__(self, layer, inp, name, feed = None):
		self.inp = inp # class = tfop
		self.out = None # class = tf.Tensor
		self.lay = layer
		self.sig = name
		self.action = ''
		self.wrap(feed)
		self.forward()

	def __call__(self, verbalise = True):
		if verbalise: self.verbalise()
		return self

	def wrap(self, feed):
		"""
		wraps `self.lay` into tf variables & placeholders
		if layer does not carry value (partial net loaded)
		the corresponding tf variable will be initialised 
		""" 
		if feed is None: return
		for var in self.lay.wshape: # trainable vars
			self.action = 'load'
			sig = '{}-{}'.format(self.sig, var) 
			val = self.lay.w.get(var, None)
			if val is None: # darknet is partially loaded
				args = [self.lay.wshape[var], 0., 1e-2]
				val = tf.truncated_normal(*args)
				self.action = 'init '

			self.lay.w[var] = tf.Variable(val, name = sig)
		
		for ph in self.lay.h: # placeholders
			sig = '{}-{}'.format(self.sig, ph)
			values = self.lay.h[ph]; shp = values['shape']
			self.lay.h[ph] = tf.placeholder(tf.float32, shp, sig)
			feed[self.lay.h[ph]] = values

	def verbalise(self):
		form = '{:<5}{:<40} -> {}' # verbalise template
		if self.inp.out.name.split(':')[0] == 'input': \
		print form.format('', 'Input size', _shape(self.inp.out))
		print form.format(self.action, self.speak(), _shape(self.out))
	
	def speak(self): pass

class convolutional(tfop):
	def forward(self):
		pad = [[self.lay.pad, self.lay.pad]] * 2;
		temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])
		
		temp = tf.nn.conv2d(temp, self.lay.w['kernel'], padding = 'VALID', 
	        name = self.sig, strides = [1] + [self.lay.stride] * 2 + [1])

		if self.lay.batch_norm: temp = self.batchnorm(self.lay, temp)
		self.out = tf.nn.bias_add(temp, self.lay.w['biases'])

	def batchnorm(self, l, x):
		return tf.nn.batch_normalization(
			x = x, mean = l.w['mean'], offset = None, 
			variance = l.w['var'], scale = l.w['scale'], 
			variance_epsilon = 1e-5, name = self.sig + '-bnorm'
		)

	def speak(self):
		msg = 'conv{}'.format(_shape(self.lay.w['kernel']))
		return '{:<23} pad {:<2}  {}'.format(msg, 
			self.lay.pad, self.lay.batch_norm * '+bnorm')


"""	
Simpler ops:
full, flatten, maxpool, avgpool, leaky, dropout
"""

class connected(tfop):
	def forward(self):
		self.out = tf.nn.xw_plus_b(
			self.inp.out,
			self.lay.w['weights'], 
			self.lay.w['biases'], 
			name = self.sig)

	def speak(self):
		return 'full{}'.format(_shape(self.lay.w['weights']))

class flatten(tfop):
	def forward(self):
		temp = tf.transpose(self.inp.out, [0,3,1,2])
		self.out = slim.flatten(temp, scope = self.sig)

	def speak(self): return 'flat()'

class softmax(tfop):
	def forward(self):
		self.out = tf.nn.softmax(self.inp.out)

	def speak(self): return 'softmax()'

class avgpool(tfop):
	def forward(self):
		self.out = tf.reduce_mean(
			self.inp.out, [1, 2], 
			name = self.sig
		)

	def speak(self): return 'avgpool()'

class dropout(tfop):
	def forward(self):
		self.out = tf.nn.dropout(
			self.inp.out, 
			self.lay.h['pdrop'], 
			name = self.sig
		)

	def speak(self): return 'drop()'

class crop(tfop):
	def forward(self):
		self.out =  self.inp.out * 2. - 1.
		
	def verbalise(self): pass

class maxpool(tfop):
	def forward(self):
		self.out = tf.nn.max_pool(
			self.inp.out, padding = 'VALID',
	        ksize = [1] + [self.lay.ksize]*2 + [1], 
	        strides = [1] + [self.lay.stride]*2 + [1],
	        name = self.sig
	    )
	
	def verbalise(self): pass

class leaky(tfop):
	def forward(self):
		self.out = tf.maximum(
			.1 * self.inp.out, 
			self.inp.out, 
			name = self.sig
		)

	def verbalise(self): pass


class identity(tfop):
	def __init__(self, inp):
		self.inp = None
		self.out = inp

op_types = {
	'convolutional': convolutional,
	'connected': connected,
	'maxpool': maxpool,
	'leaky': leaky,
	'dropout': dropout,
	'flatten': flatten,
	'avgpool': avgpool,
	'softmax': softmax,
	'identity': identity,
	'crop': crop
}

def op_create(*args):
	layer_type = list(args)[0].type
	return op_types[layer_type](*args)