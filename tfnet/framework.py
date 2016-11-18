"""
framework factory
"""

from yolo import *

class framework(object):
	def __init__(self):
		pass

class yolo(framework):
	def __init__(self):
		self.loss = yolo_loss
		self.parse = yolo_parse
		self.batch = yolo_batch
		self.metaprocess = yolo_metaprocess
		self.preprocess = yolo_preprocess
		self.postprocess = yolo_postprocess
		self.is_inp = is_yolo_inp

types = {
	'[detection]': yolo
}

def create_framework(net_type):
	return types.get(net_type, framework)()