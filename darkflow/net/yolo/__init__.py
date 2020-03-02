from . import train
from . import predict
from . import data
from . import misc
import numpy as np


""" YOLO framework __init__ equivalent"""

##using misc.labels to assign labels and check if the no of labels are consistent with the model output classes and asigning colors to each labels

def constructor(self, meta, FLAGS):

	def _to_color(indx, base): #ignore for now
		""" return (b, r, g) tuple"""
		base2 = base * base
		b = 2 - indx / base2
		r = 2 - (indx % base2) / base
		g = 2 - (indx % base2) % base
		return (b * 127, r * 127, g * 127)
	if 'labels' not in meta: #if no labels
		misc.labels(meta, FLAGS) #get labels #We're not loading from a .pb so we do need to load the labels
	assert len(meta['labels']) == meta['classes'], ( #check if length of our labels is equal to the number of classes
		'labels.txt and {} indicate' + ' ' #inconsistency
		'inconsistent class numbers'
	).format(meta['model'])

	# assign a color for each label
	colors = list() 
	base = int(np.ceil(pow(meta['classes'], 1./3)))
	for x in range(len(meta['labels'])): 
		colors += [_to_color(x, base)] #generating unique RGB per label
	meta['colors'] = colors
	self.fetch = list()
	self.meta, self.FLAGS = meta, FLAGS

	# over-ride the threshold in meta if FLAGS has it.
	if FLAGS.threshold > 0.0:
		self.meta['thresh'] = FLAGS.threshold