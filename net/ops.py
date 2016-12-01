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

		self.convert(feed)
		self.forward()

	def __call__(self, verbalise = True):
		if verbalise: self.verbalise()
		return self

	def convert(self, feed):
		"""
		convert `self.lay` into variables & placeholders
		some of the variables 
		""" 
		if feed is None: return

		delegate_vars = ['gamma', 'moving_mean', 'moving_variance']

		self.action = None
		for var in self.lay.wshape: # variables
			val = self.lay.w.get(var, None)

			if val is None:
				args = [self.lay.wshape[var], 0., 1e-2]
				self.lay.w[var] = tf.truncated_normal(*args)
				self.action = 'Init'
			else:
				self.lay.w[var] = tf.constant_initializer(val)
				self.action = 'Load'

			if var not in delegate_vars:
				with tf.variable_scope(self.sig):
					self.lay.w[var] = tf.get_variable(var,
						shape = self.lay.wshape[var],
						dtype = tf.float32,
						initializer = self.lay.w[var])
		
		for ph in self.lay.h: # placeholders
			sig = '{}-{}'.format(self.sig, ph)
			val = self.lay.h[ph] 
			shp = val['shape']
			dft = val['dfault']

			self.lay.h[ph] = tf.placeholder_with_default(
				val['dfault'], val['shape'], name = sig)
			feed[self.lay.h[ph]] = val['feed']

	def verbalise(self):
		form = '{:<4} {:<43} -> {}' # verbalise template
		nothing = '----'

		if self.inp.out.name.split(':')[0] == 'input': \
		print form.format(nothing, 'input size', _shape(self.inp.out))
		
		if self.action is None: self.action = nothing
		if not self.action: return
		print form.format(self.action, self.speak(), _shape(self.out))
	
	def speak(self): pass

class local(tfop):
	def forward(self):
		pad = [[self.lay.pad, self.lay.pad]] * 2;
		temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])

		k = self.lay.w['kernels']
		ksz = self.lay.ksize
		half = ksz/2
		out = list()
		for i in range(self.lay.h_out):
			row_i = list()
			for j in range(self.lay.w_out):
				kij = k[i * self.lay.w_out + j]
				i_, j_ = i + 1 - half, j + 1 - half
				tij = temp[:, i_ : i_ + ksz, j_ : j_ + ksz,:]
				row_i.append(
					tf.nn.conv2d(tij, kij, 
						padding = 'VALID', 
						strides = [1] * 4)
				)
			out += [tf.concat(2, row_i)]

		self.out = tf.concat(1, out)

	def speak(self):
		msg = 'loca{}'.format(_shape(self.lay.w['kernels']))
		return '{:<26} pad {:<2}'.format(msg, 
			self.lay.pad * self.lay.ksize / 2)

class convolutional(tfop):
	def forward(self):
		pad = [[self.lay.pad, self.lay.pad]] * 2;
		temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])
		
		temp = tf.nn.conv2d(temp, self.lay.w['kernel'], padding = 'VALID', 
	        name = self.sig, strides = [1] + [self.lay.stride] * 2 + [1])

		if self.lay.batch_norm: 
			temp = self.batchnorm(self.lay, temp)
		self.out = tf.nn.bias_add(temp, self.lay.w['biases'])

	def batchnorm(self, layer, inp):
		if type(layer.h['is_training']) is bool:
			temp = (inp - layer.w['moving_mean'])
			temp /= (np.sqrt(layer.w['moving_variance']) + 1e-5)
			temp *= layer.w['gamma']
			return temp
		else: return slim.batch_norm(inp, 
			center = False, scale = True, epsilon = 1e-5,
			initializers = layer.w, scope = self.sig,
			is_training = layer.h['is_training'])

	def speak(self):
		msg = 'conv{}'.format(_shape(self.lay.w['kernel']))
		return '{:<26} pad {:<2}  {}'.format(msg, 
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
		self.action = False

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
	'crop': crop,
	'local': local
}

def op_create(*args):
	layer_type = list(args)[0].type
	return op_types[layer_type](*args)