import tensorflow as tf
import time
import flow
import help
from ops import op_create, identity
from ops import HEADER, LINE
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
	say = help.say
	train = flow.train
	shuffle = help.shuffle
	predict = flow.predict
	build_train_op = help.build_train_op
	load_from_ckpt = help.load_from_ckpt

	def __init__(self, FLAGS, darknet = None):
		if darknet is None:	darknet = Darknet(FLAGS)
		self.framework = create_framework(darknet.meta['type'])
		self.meta = self.framework.metaprocess(darknet.meta)
		self.num_layer = len(darknet.layers)
		self.darknet = darknet
		self.FLAGS = FLAGS

		self.say('\nBuilding net ...')
		start = time.time()
		self.graph = tf.Graph()
		with self.graph.as_default() as g:
			self.build_forward()
			self.setup_meta_ops()
		self.say('Finished in {}s\n'.format(
			time.time() - start))


	def build_forward(self):
		verbalise = self.FLAGS.verbalise
		ntrain = self.FLAGS.train
		if ntrain < 0: # train all
			ntrain = self.num_layer

		# Placeholders
		inp_size = [None] + self.meta['inp_size']
		self.inp = tf.placeholder(tf.float32, inp_size, 'input')
		self.feed = dict() # other placeholders

		# Build the forward pass
		state = identity(self.inp)
		roof = self.num_layer - ntrain
		self.say(HEADER, LINE)
		for i, layer in enumerate(self.darknet.layers):
			scope = '{}-{}'.format(str(i),layer.type)
			args = [layer, state, i, roof, self.feed]
			state = op_create(*args)
			mess = state.verbalise()
			self.say(mess)
		self.say(LINE)

		self.top = state
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

		if self.FLAGS.train == 0: return
		self.saver = tf.train.Saver(tf.all_variables(), 
			max_to_keep = self.FLAGS.keep)
		if self.FLAGS.load != 0: self.load_from_ckpt()

	def savepb(self):
		"""
		Create a standalone const graph def that 
		C++	can load and run.
		"""
		darknet_pb = self.to_darknet()
		flags_pb = self.FLAGS
		flags_pb.train = 0
		flags_pb.verbalise = False
		
		# rebuild another tfnet. all const.
		tfnet_pb = TFNet(flags_pb, darknet_pb)		
		tfnet_pb.sess = tf.Session(graph = tfnet_pb.graph)
		#tfnet_pb.predict() # uncomment for unit testing
		name = 'graph-{}.pb'.format(self.meta['name'])
		self.say('Saving const graph def to {}'.format(name))
		graph_def = tfnet_pb.sess.graph_def
		tf.train.write_graph(graph_def,'./',name,False)