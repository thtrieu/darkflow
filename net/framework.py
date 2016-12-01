"""
framework factory
"""
import tensorflow as tf
import yolo

class framework(object):
	def __init__(self):
		pass
	def metaprocess(self, meta):
		return meta
	def is_inp(self, inp):
		return True
	def loss(self, net):
		m = net.meta
		_truth = tf.placeholder(tf.float32, net.out.get_shape())
		placeholders = {
			'truth': _truth
		}
		loss = tf.nn.l2_loss(_truth - net.out)
		return placeholders, loss

class YOLO(framework):
	def __init__(self):
		self.loss = yolo.train.loss
		self.parse = yolo.train.parse
		self.batch = yolo.train.batch
		self.metaprocess = yolo.test.metaprocess
		self.preprocess = yolo.test.preprocess
		self.postprocess = yolo.test.postprocess
		self.is_inp = yolo.misc.is_inp

types = {
	'[detection]': YOLO
}

def create_framework(net_type):
	return types.get(net_type, framework)()