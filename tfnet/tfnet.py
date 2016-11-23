"""
file: tfnet.py
includes: definition of class TFNet
this class initializes by building the forward pass
its methods include train, predict and savepb - saving
the current model to a protobuf file (no variable included)
"""

import tensorflow as tf
from tfop import op_create, identity
from tfnet_flow import tf_train, tf_predict
from tfnet_help import tf_build_train_op, tf_load_from_ckpt
from framework import create_framework

class TFNet(object):

	_TRAINER = dict({
		'rmsprop': tf.train.RMSPropOptimizer,
		'adadelta': tf.train.AdadeltaOptimizer,
		'adagrad': tf.train.AdagradOptimizer,
		'adagradDA': tf.train.AdagradDAOptimizer,
		'momentum': tf.train.MomentumOptimizer,
		'adam': tf.train.AdamOptimizer,
		'ftrl': tf.train.FtrlOptimizer,
	})

	# imported methods
	train = tf_train
	predict = tf_predict
	build_train_op = tf_build_train_op
	load_from_ckpt = tf_load_from_ckpt

	def __init__(self, darknet, FLAGS):
		self.framework = create_framework(darknet.meta['type'])
		self.meta = self.framework.metaprocess(darknet.meta)
		self.darknet = darknet
		self.FLAGS = FLAGS

		self.graph = tf.Graph()
		with self.graph.as_default() as g:
			self.build_forward()
			self.setup_meta_ops()


	def build_forward(self):
		self.ckpt = self.FLAGS.savepb is None
		verbalise = self.FLAGS.verbalise

		# Placeholder
		inp_size = [None] + self.meta['inp_size']
		self.inp = tf.placeholder(tf.float32, inp_size, 'input')
		self.feed = dict() # other placeholders

		# Build the forward pass
		now = identity(self.inp)
		num = len(self.darknet.layers)
		for i, layer in enumerate(self.darknet.layers):
			name = '{}-{}'.format(str(i),layer.type)
			args = [layer, now, name]
			if not self.ckpt: args += [self.feed]
			now = op_create(*args)(verbalise)
		self.top = now

		# Attach the now.out to self
		self.out = tf.identity(now.out, name='output')

	def setup_meta_ops(self):
		cfg = {
			'allow_soft_placement': False,
			'log_device_placement': False}
		if self.FLAGS.gpu > 0: 
			utility = min(self.FLAGS.gpu, 1.)
			print 'GPU mode with {} usage'.format(utility)
			cfg['gpu_options'] = tf.GPUOptions(
				per_process_gpu_memory_fraction = utility)
			cfg['allow_soft_placement'] = True

		self.sess = tf.Session(config = tf.ConfigProto(**cfg))
		if self.FLAGS.train: self.build_train_op()
		self.sess.run(tf.initialize_all_variables())
		if self.ckpt: return
		self.saver = tf.train.Saver(tf.all_variables(), 
			max_to_keep = self.FLAGS.keep)
		if self.FLAGS.load != 0: self.load_from_ckpt()

	def savepb(self):
		"""
		Create a standalone const graph def, so that 
		C++	can load and run it. What's good abt it?
		1. Don't double the necessary size
		2. Convert on the fly - at any point you want
		"""
		darknet_ckpt = to_darknet(self)
		flags_ckpt = self.FLAGS
		flags_ckpt.savepb = None # signal
		flags_ckpt.verbalise = False

		# placeholder takes default vals
		for layer in darknet_ckpt.layers:
			for ph in layer.h:
				layer.h[ph] =  layer.h[ph]['dfault']
		
		# rebuild another tfnet. all const.
		tfnet_ckpt = TFNet(darknet_ckpt, flags_ckpt)		
		tfnet_ckpt.sess = tf.Session(graph = tfnet_ckpt.graph)
		# tfnet_ckpt.predict() # uncomment for unit testing
		name = 'graph-{}.pb'.format(self.meta['model'])
		print 'Saving const graph def to {}'.format(name)
		graph_def = tfnet_ckpt.sess.graph_def
		tf.train.write_graph(graph_def,'./',name,False)