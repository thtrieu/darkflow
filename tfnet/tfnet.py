"""
file: tfnet.py
includes: definition of class TFNet
this class initializes by building the forward pass
its methods include train, predict and savepb - saving
the current model to a protobuf file (no variable included)
"""

from tfnet_flow import *
from darknet import *
from tfop import *

class TFNet(object):

	# imported methods
	train = tf_train
	predict = tf_predict
	shuffle = tf_shuffle
	to_darknet = to_darknet
	load_old_graph = load_old_graph

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
		self.feed = dict() # for any other placeholders, e.g. dropout

		# Iterate through darknet layers
		now = self.inp
		num = len(self.darknet.layers)
		for i, layer in enumerate(self.darknet.layers):
			name = '{}-{}'.format(str(i),layer.type)
			if i+1 == num: name += '-tfnetoutput'
			args = [layer, now, name]
			if not self.ckpt: args += [self.feed]
			now = op_create(*args)(verbalise)

		# Attach the output to this tfnet
		self.out = now

	def setup_meta_ops(self):
		cfg = {
			'allow_soft_placement': False,
			'log_device_placement': False}
		if self.FLAGS.gpu > 0: 
			utility = min(FLAGS.gpu, 1.)
			print 'GPU mode with {} usage'.format(utility)
			cfg['gpu_options'] = tf.GPUOptions(
				per_process_gpu_memory_fraction = utility)
			cfg['allow_soft_placement'] = True

		self.sess = tf.Session(config = tf.ConfigProto(**cfg))
		if self.FLAGS.train:
			loss_return = self.framework.loss(self)
			self.placeholders, self.loss, self.train_op = loss_return
		self.sess.run(tf.initialize_all_variables())
		
		if self.ckpt: return
		self.saver = tf.train.Saver(tf.all_variables(), 
			max_to_keep = self.FLAGS.keep)

		if self.FLAGS.load == 0: return
		if self.FLAGS.load < 0:
			with open('backup/checkpoint', 'r') as f:
				last = f.readlines()[-1].strip()
				load_point = last.split(' ')[1]
				load_point = load_point.split('"')[1]
				load_point = load_point.split('-')[-1]
				self.FLAGS.load = int(load_point)
		
		load_point = 'backup/{}'.format(self.meta['model'])
		load_point = '{}-{}'.format(load_point, self.FLAGS.load)		
		print 'Loading from {}'.format(load_point)
		try: self.saver.restore(self.sess, load_point)
		except: self.load_old_graph(load_point)

	def savepb(self):
		"""
		Create a standalone const graph def
		So that C++ can load and run it.
		What's good abt it?
		1. Don't double the necessary size
		2. Convert on the fly - at any point you want
		"""
		darknet_ckpt = self.to_darknet()
		flags_ckpt = self.FLAGS
		flags_ckpt.savepb = None # signal
		flags_ckpt.verbalise = False

		# placeholder takes default vals
		for layer in darknet_ckpt.layers:
			for ph in layer.h:
				layer.h[ph] =  layer.h[ph]['dfault']
		
		tfnet_ckpt = TFNet(darknet_ckpt, flags_ckpt)		
		tfnet_ckpt.sess = tf.Session(graph = tfnet_ckpt.graph)
		# tfnet_ckpt.predict() # uncomment for unit testing
		name = 'graph-{}.pb'.format(self.meta['model'])
		print 'Saving const graph def to {}'.format(name)
		graph_def = tfnet_ckpt.sess.graph_def
		tf.train.write_graph(graph_def,'./',name,False)