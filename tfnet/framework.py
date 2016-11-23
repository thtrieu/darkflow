"""
framework factory
"""

import yolo

class framework(object):
	def __init__(self):
		pass

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