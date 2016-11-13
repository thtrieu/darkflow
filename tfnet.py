"""
file: tfnet.py
includes: definition of class TFNet
this class initializes by building the forward pass
its methods include train, predict and savepb - saving
the current model to a protobuf file (no variable included)
"""

import sys
from yolo.drawer import *
from darknet import *
from ops import *
from data import *

const_layer = ['leaky', 'dropout']
var_layer = ['convolutional', 'connected']

class TFNet(object):
	def __init__(self, darknet, FLAGS):
		# Attach model's hyper params to the tfnet
		self.meta = yolo_metaprocess(darknet.meta)
		self.FLAGS = FLAGS	

		# Placeholders
		self.inp = tf.placeholder(tf.float32,
			[None, 448, 448, 3], name = 'input')
		self.drop = dict()
		self.feed = dict()

		# Iterate through darknet layers
		now = self.inp
		for i, l in enumerate(darknet.layers):
			if i == len(darknet.layers)-1: name = 'output'
			else: name = l.type+'-{}'.format(i)
			# no variable when saving to .pb file
			if l.type in var_layer and not FLAGS.savepb:
				for var in l.p: l.p[var] = tf.Variable(l.p[var])
			arg = [l, now, name]
			if l.type=='convolutional': now = convl(*arg)
			elif l.type == 'connected': now = dense(*arg)
			elif l.type == 'maxpool': now = maxpool(*arg)	
			elif l.type == 'flatten': now = flatten(*arg[1:])
			elif l.type == 'leaky'  : now =   leaky(*arg[1:])
			# Dropout
			elif l.type == 'dropout' and not FLAGS.savepb:
				print 'Dropout p = {}'.format(l.prob)
				self.drop[name] = tf.placeholder(tf.float32)
				drop_value_name = '{}_'.format(name)
				self.drop[drop_value_name] = l.prob
				self.feed[self.drop[name]] = self.drop[drop_value_name]
				now = dropout(now, self.drop[name], name)
			if l.type not in const_layer: print now.get_shape()

		# Attach the output to this tfnet
		self.out = now

	def setup_meta_ops(self):
		if self.FLAGS.gpu > 0: 
			percentage = min(FLAGS.gpu, 1.)
			print 'GPU mode with {} usage'.format(percentage)
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=percentage)
			self.sess = tf.Session(config = tf.ConfigProto(
				allow_soft_placement = True,
				log_device_placement = False,
				gpu_options = gpu_options))
		else:
			print 'CPU mode'
			self.sess = tf.Session(config = tf.ConfigProto(
				allow_soft_placement = False,
				log_device_placement = False))
		if self.FLAGS.train: yolo_loss(self)
		if self.FLAGS.savepb: 
			self.savepb('graph-{}.pb'.format(self.meta['model']))
			sys.exit()
		else: self.saver = tf.train.Saver(tf.all_variables(), 
			max_to_keep = self.FLAGS.keep)
		self.sess.run(tf.initialize_all_variables())
		if self.FLAGS.load > 0:
			load_point = 'backup/model-{}'.format(self.step)
			print 'Loading from {}'.format(load_point)
			self.saver.restore(self.sess, load_point)

	def savepb(self, name):
		print 'Saving pb to {}'.format(name)
		tf.train.write_graph(self.sess.graph_def,'./', name, as_text = False)

	def to_constant(self, inc = 0):
		with open('binaries/yolo-{}-{}.weights'.format(
			self.meta['model'].split('-')[0], self.step + inc), 'w') as f:
			f.write(np.array([1]*4, dtype=np.int32).tobytes())
			for i, variable in enumerate(tf.trainable_variables()):
				val = variable.eval(self.sess)
				if len(val.shape) == 4:
					val = val.transpose([3,2,0,1])
				val = val.reshape([-1])
				f.write(val.tobytes())
	
	def train(self, train_set, parsed_annota, batch, epoch):
		batches = shuffle(train_set, parsed_annota, batch, epoch, self.meta)

		print 'Training statistics:'
		print '   Learning rate : {}'.format(self.FLAGS.lr)
		print '   Batch size    : {}'.format(batch)
		print '   Epoch number  : {}'.format(epoch)
		print '   Backup every  : {}'.format(self.FLAGS.save)

		total = int() # total number of batches
		for i, packet in enumerate(batches):
			if i == 0: total = packet; continue
			x_batch, datum = packet
			feed_dict = yolo_feed_dict(self, x_batch, datum)
			feed_dict[self.inp] = x_batch
			for k in self.feed: feed_dict[k] = self.feed[k]

			_, loss = self.sess.run([self.train_op, self.loss], feed_dict)
			print 'step {} - batch {} - loss {}'.format(i+self.step, i, loss)
			if i % (self.FLAGS.save/batch) == 0 or i == total:
				print 'save checkpoint and binaries at step {}'.format(self.step+i)
				self.saver.save(self.sess, 'backup/model-{}'.format(self.step+i))
				self.to_constant(inc = i)

	def predict(self):
		inp_path = self.FLAGS.testset
		all_inp_ = os.listdir(inp_path)
		all_inp_ = [i for i in all_inp_ if is_yolo_inp(i)]
		batch = min(self.FLAGS.batch, len(all_inp_))

		for j in range(len(all_inp_)/batch):
			inp_feed = list()
			all_inp = all_inp_[j*batch: (j*batch+batch)]
			new_all = list()
			for inp in all_inp:
				new_all += [inp]
				this_inp = '{}/{}'.format(inp_path, inp)
				this_inp = yolo_preprocess(this_inp)
				inp_feed.append(this_inp)
			all_inp = new_all

			feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
			for k in self.feed: feed_dict[k] = 1.0
		
			print ('Forwarding {} inputs ...'.format(len(inp_feed)))
			start = time.time()
			out = self.sess.run([self.out], feed_dict)
			stop = time.time()
			last = stop - start
			print ('Total time = {}s / {} inps = {} ips'.format(
				last, len(inp_feed), len(inp_feed) / last))

			for i, prediction in enumerate(out[0]):
				yolo_postprocess(
					prediction, '{}/{}'.format(inp_path, all_inp[i]), 
					self.FLAGS, self.meta)
