"""
file: tfnet.py
includes: definition of class TFNet
this class initializes by building the forward pass
its methods include train, predict and savepb - saving
the current model to a protobuf file (no variable included)
"""

import tensorflow as tf
import time
from ops import op_create, identity
from flow import tf_train, tf_predict
from help import tf_build_train_op, tf_load_from_ckpt, tf_say, to_darknet
from framework import create_framework
from dark.darknet import Darknet

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
	say = tf_say
	train = tf_train
	predict = tf_predict
	build_train_op = tf_build_train_op
	load_from_ckpt = tf_load_from_ckpt

	def __init__(self, FLAGS, darknet = None):
		if darknet is None:	darknet = Darknet(FLAGS)
		self.framework = create_framework(darknet.meta['type'])
		self.meta = self.framework.metaprocess(darknet.meta)
		self.darknet = darknet
		self.FLAGS = FLAGS

		self.say('\nCompiling net & fill in parameters...')
		start = time.time()
		self.graph = tf.Graph()
		with self.graph.as_default() as g:
			self.build_forward()
			self.setup_meta_ops()
		self.say('Finished in {}s\n'.format(time.time() - start))


	def build_forward(self):
		const = self.FLAGS.savepb == 'saving'
		verbalise = self.FLAGS.verbalise

		# Placeholder
		inp_size = [None] + self.meta['inp_size']
		self.inp = tf.placeholder(tf.float32, inp_size, 'input')
		self.feed = dict() # other placeholders

		# Build the forward pass
		state = identity(self.inp)
		num = len(self.darknet.layers)
		for i, layer in enumerate(self.darknet.layers):
			name = '{}-{}'.format(str(i),layer.type)
			args = [layer, state, name]
			if not const: args += [self.feed]
			state = op_create(*args)(verbalise)
		self.top = state

		# Attach the state.out to self
		self.out = tf.identity(state.out, name='output')

	def setup_meta_ops(self):
		cfg = dict({
			'allow_soft_placement': False,
			'log_device_placement': False
		})

		utility = min(self.FLAGS.gpu, 1.)
		if utility > 0.0:
			self.say('GPU mode with {} usage'.format(utility))
			cfg['gpu_options'] = tf.GPUOptions(
				per_process_gpu_memory_fraction = utility)
			cfg['allow_soft_placement'] = True
		else: 
			self.say('Running entirely on CPU')
			cfg['device_count'] = {'GPU': 0}

		if self.FLAGS.savepb == 'saving': return
		if self.FLAGS.train: self.build_train_op()
		self.sess = tf.Session(config = tf.ConfigProto(**cfg))
		self.sess.run(tf.initialize_all_variables())

		self.saver = tf.train.Saver(tf.all_variables(), 
			max_to_keep = self.FLAGS.keep)
		if self.FLAGS.load != 0: self.load_from_ckpt()

	def savepb(self):
		"""
		Create a standalone const graph def that 
		C++	can load and run.
		"""
		darknet_pb = to_darknet(self)
		flags_pb = self.FLAGS
		flags_pb.savepb = 'saving' # signal
		flags_pb.verbalise = False
		
		# rebuild another tfnet. all const.
		tfnet_pb = TFNet(flags_pb, darknet_pb)		
		tfnet_pb.sess = tf.Session(graph = tfnet_pb.graph)
		tfnet_pb.predict() # uncomment for unit testing
		name = 'graph-{}.pb'.format(self.meta['name'])
		self.say('Saving const graph def to {}'.format(name))
		graph_def = tfnet_pb.sess.graph_def
		tf.train.write_graph(graph_def,'./',name,False)