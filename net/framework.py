import yolo
import yolov2

class framework(object):
	def __init__(self, *args):
		self.constructor(*args)

class YOLO(framework):
	constructor = yolo.constructor
	loss = yolo.train.loss
	parse = yolo.train.parse
	batch = yolo.train.batch
	preprocess = yolo.test.preprocess
	postprocess = yolo.test.postprocess
	is_inp = yolo.misc.is_inp
	profile = yolo.misc.profile

class YOLOv2(framework):
	constructor = yolo.constructor
	preprocess = yolo.train.preprocess
	parse = yolo.train.parse
	# self.batch
	# self.loss
	postprocess = yolov2.test.postprocess
	is_inp = yolo.misc.is_inp

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