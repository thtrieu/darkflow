"""
tfnet secondary (helper) methods
"""

from numpy.random import permutation as perm
from utils.loader import create_loader
import tensorflow as tf
import numpy as np
import os

old_graph_msg = 'Resolving old graph def {} (no guarantee)'

def build_train_op(self):
	loss_ops = self.framework.loss(self.out)
	self.placeholders, self.loss = loss_ops

	self.say('Building {} train op'.format(self.meta['model']))
	optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
	gradients = optimizer.compute_gradients(self.loss)
	self.train_op = optimizer.apply_gradients(gradients)

def load_from_ckpt(self):
	if self.FLAGS.load < 0: # load lastest ckpt
		with open(self.FLAGS.backup + 'checkpoint', 'r') as f:
			last = f.readlines()[-1].strip()
			load_point = last.split(' ')[1]
			load_point = load_point.split('"')[1]
			load_point = load_point.split('-')[-1]
			self.FLAGS.load = int(load_point)
	
	load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
	load_point = '{}-{}'.format(load_point, self.FLAGS.load)
	self.say('Loading from {}'.format(load_point))
	try: self.saver.restore(self.sess, load_point)
	except: load_old_graph(self, load_point)

def say(self, *msgs):
	if not self.FLAGS.verbalise:
		return
	msgs = list(msgs)
	for msg in msgs:
		if msg is None: continue
		print msg

def shuffle(self):
	"""
	Call the specific framework to parse annotations, then use the parsed 
	object to yield minibatches. minibatches should be preprocessed before
	yielding to be appropriate placeholders for model's loss evaluation.
	"""
	data = self.framework.parse()
	size = len(data); batch = self.FLAGS.batch

	self.say('Dataset of {} instance(s)'.format(size))
	if batch > size: 
		self.FLAGS.batch = batch = size
	batch_per_epoch = int(size / batch)
	total = self.FLAGS.epoch * batch_per_epoch
	yield total

	for i in range(self.FLAGS.epoch):
		self.say('EPOCH {}'.format(i + 1))
		shuffle_idx = perm(np.arange(size))
		for b in range(batch_per_epoch): 
			end_idx = (b+1) * batch
			start_idx = b * batch
			# two yieldee
			x_batch = list()
			feed_batch = dict()

			for j in range(start_idx, end_idx):
				real_idx = shuffle_idx[j]
				this = data[real_idx]
				inp, feedval = self.framework.batch(this)
				if inp is None: continue

				x_batch += [np.expand_dims(inp, 0)]
				for key in feedval:
					if key not in feed_batch: 
						feed_batch[key] = [feedval[key]]; 
						continue
					feed_batch[key] = np.concatenate(
						[feed_batch[key], [feedval[key]]])		
			
			x_batch = np.concatenate(x_batch, 0)
			yield (x_batch, feed_batch)


def load_old_graph(self, ckpt):	
	ckpt_loader = create_loader(ckpt)
	self.say(old_graph_msg.format(ckpt))
	
	for var in tf.all_variables():
		name = var.name.split(':')[0]
		args = [name, var.get_shape()]
		val = ckpt_loader(args)
		assert val is not None, \
		'Cannot find and load {}'.format(var.name)
		shp = val.shape
		plh = tf.placeholder(tf.float32, shp)
		op = tf.assign(var, plh)
		self.sess.run(op, {plh: val})


def to_darknet(self):
	"""
	Translate from TFNet back to darknet
	"""
	darknet_ckpt = self.darknet
	with self.graph.as_default() as g:
		for var in tf.all_variables():
			name = var.name.split(':')[0]
			var_name = name.split('-')
			l_idx = int(var_name[0])
			w_sig = var_name[1].split('/')[-1]
			l = darknet_ckpt.layers[l_idx]
			l.w[w_sig] = var.eval(self.sess)

	for layer in darknet_ckpt.layers:
		for ph in layer.h:
			feed = self.feed[layer.h[ph]]
			layer.h[ph] = feed

	return darknet_ckpt