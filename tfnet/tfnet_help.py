"""
tfnet secondary (helper) methods
"""

import tensorflow as tf
import numpy as np

off_bound_msg = 'Random scale/translate sends obj off bound'
too_big_batch = 'Batch size is bigger than training data size'
old_graph_msg = 'Resolving incompatible graph definition'

def tf_shuffle(self):
	"""
	Call the specific to parse annotations, where or not doing the parse
	is up to the model. Then use the parsed object to yield minibatches
	minibatches will be preprocessed before yielding to be appropriate
	placeholders for model's loss evaluation.
	"""
	data = self.framework.parse(self.FLAGS, self.meta)
	size = len(data); batch = self.FLAGS.batch

	print 'Dataset of {} instance(s)'.format(size)
	assert batch <= size, too_big_batch
	batch_per_epoch = int(size / batch)
	total = self.FLAGS.epoch * batch_per_epoch
	yield total

	for i in range(self.FLAGS.epoch):
		print 'EPOCH {}'.format(i+1)
		shuffle_idx = np.random.permutation(np.arange(size))
		for b in range(batch_per_epoch):
			start_idx = b * batch
			end_idx = (b+1) * batch

			datum = dict()
			x_batch = list()
			offbound = False
			for j in range(start_idx,end_idx):
				real_idx = shuffle_idx[j]
				this = data[real_idx]
				inp, tensors = self.framework.batch(
					self.FLAGS, self.meta, this)
				if inp is None: offbound = True; break
				x_batch += [inp]
				for k in tensors:
					if k not in datum: datum[k] = [tensors[k]]; continue
					datum[k] = np.concatenate([datum[k], [tensors[k]]])		
			
			if offbound: print off_bound_msg; continue
			x_batch = np.concatenate(x_batch, 0)
			yield (x_batch, datum)

def load_old_graph(self, load_point):
	"""
	new versions of code name variables differently,
	so for backward compatibility, a matching between
	new and old graph_def has to be done.
	"""
	def lookup(allw, name, shape):
		"""
		Look for variable with name `name`
		and shape `shape` in list `allw`
		"""
		for idx, w in enumerate(allw):
			if w.name == name: # highly unlikely
				return idx	
		for idx, w in enumerate(allw):
			if w.get_shape() == shape:
				return idx
		return None

	print old_graph_msg
	meta = '{}.meta'.format(load_point)
	msg = 'Recovery from {} '.format(meta)
	err = '{}'.format(msg + 'failed')

	feed = dict()
	allw = tf.all_variables()
	with tf.Graph().as_default() as g:
		with tf.Session().as_default() as sess:
			old_meta = tf.train.import_meta_graph(meta)
			old_meta.restore(sess, load_point)
			for i, this in enumerate(tf.all_variables()):
				name = this.name
				shape = this.get_shape()
				args = [allw, name, shape]
				idx = lookup(*args)
				if idx is None: continue
				val = this.eval(sess)
				feed[allw[idx]] = val
				del allw[idx]

	# Make sure all new vars are covered
	assert allw == list(), err
	
	# restore values
	for w in feed:
		val = feed[w]; shape = val.shape
		ph = tf.placeholder(tf.float32, shape)
		op = tf.assign(w, ph) # use placeholder
		self.sess.run(op, {ph: val})


def to_darknet(self):
	"""
	Translate from TFNet back to darknet
	"""
	darknet_ckpt = self.darknet
	with self.graph.as_default() as g:
		for var in tf.trainable_variables():
			name = var.name.split(':')[0]
			val = var.eval(self.sess)
			var_name = name.split('-')
			l_idx = int(var_name[0])
			w_sig = var_name[-1]
			darknet_ckpt.layers[l_idx].w[w_sig] = val
	for layer in darknet_ckpt.layers:
		for ph in layer.h:
			feed = self.feed[layer.h[ph]]
			layer.h[ph] = feed

	return darknet_ckpt