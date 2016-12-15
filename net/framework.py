import yolo
import yolov2
import vanilla

class framework(object):
	loss = vanilla.train.loss
	def __init__(self, *args):
		self.constructor(*args)
	def is_inp(self):
		return True

class YOLO(framework):
	constructor = yolo.constructor
	parse = yolo.data.parse
	shuffle = yolo.data.shuffle
	preprocess = yolo.test.preprocess
	postprocess = yolo.test.postprocess
	loss = yolo.train.loss
	is_inp = yolo.misc.is_inp
	profile = yolo.misc.profile

class YOLOv2(framework):
	constructor = yolo.constructor
	parse = yolo.data.parse
	# shuffle = yolo.data.shuffle
	preprocess = yolo.test.preprocess
	# loss = yolo.train.loss
	is_inp = yolo.misc.is_inp
	postprocess = yolov2.test.postprocess

"""
framework factory
"""

types = {
	'[detection]': YOLO,
	'[region]': YOLOv2
}

def create_framework(meta, FLAGS):
	net_type = meta['type']
	this = types.get(net_type, framework)
	return this(meta, FLAGS)