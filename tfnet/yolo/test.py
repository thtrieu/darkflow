"""
file: yolo/drawer.py
includes: metaprocess(), preprocess() and postprocess()
together they supports two ends of the testing process:
		preprocess -> flow the net -> post process
		where flow the net is taken care of the general framework
Namely, they answers the following questions:
	0. what to prepare given `meta`, the net's hyper-parameters?
		e.g. prepare color for drawing, load labels from labels.txt
	1. what to do before flowing the net?
	2. what to do after flowing the net?
"""
from misc import labels
from utils.im_transform import im_np_recolor, imcv2_affine_trans
from utils.box import BoundBox, box_intersection, prob_compare
import numpy as np
import cv2
import os

def metaprocess(meta):
	"""
	Add to meta (a dict) `labels` correspond to current model and
	`colors` correspond to these labels, for drawing predictions.
	"""
	def to_color(indx, base):
		base2 = base * base
		b = indx / base2
		r = (indx % base2) / base
		g = (indx % base2) % base
		return (b * 127, r * 127, g * 127)
	labels(meta)
	assert len(meta['labels']) == meta['classes'], (
		'labels.txt and {} indicate' + ' '
		'inconsistent class numbers'
	).format(meta['model'])
	colors = list()
	base = int(np.ceil(pow(meta['classes'], 1./3)))
	for x in range(len(meta['labels'])): 
		colors += [to_color(x, base)]
	meta['colors'] = colors
	return meta

def preprocess(imPath, allobj = None):
	"""
	Takes an image, return it as a numpy tensor that is readily
	to be fed into tfnet. If there is an accompanied annotation (allobj),
	meaning this preprocessing is serving the train process, then this
	image will be transformed with random noise to augment training data, 
	using scale, translation, flipping and recolor. The accompanied 
	parsed annotation (allobj) will also be modified accordingly.
	"""	
	def fix(obj, dims, scale, offs):
		for i in range(1, 5):
			dim = dims[(i + 1) % 2]
			off = offs[(i + 1) % 2]
			obj[i] = int(obj[i]*scale-off)
			obj[i] = max(min(obj[i], dim), 0)
	
	im = cv2.imread(imPath)
	if allobj is not None: # in training mode
		result = imcv2_affine_trans(im)
		im, dims, trans_param = result
		scale, offs, flip = trans_param
		for obj in allobj:
			fix(obj, dims, scale, offs)
			if not flip: continue
			obj_1_ =  obj[1]
			obj[1] = dims[0] - obj[3]
			obj[3] = dims[0] - obj_1_

	size = (448, 448)
	resized = cv2.resize(im, size)
	im_np = np.array(resized)

	# recoloring as np is fast.
	if allobj is not None:
		im_np = im_np_recolor(im_np)

	# return np array input to YOLO
	im_np = im_np / 255.
	im_np = im_np * 2. - 1.
	im_np = np.expand_dims(im_np, 0)
	if allobj is None: return im_np
	return im_np #, im_ # for unit testing
	

def postprocess(predictions, img_path, FLAGS, meta):
	"""
	Takes net output, draw predictions, save to results/
	prediction is a numpy tensor - net's output
	img_path is the path to testing folder
	FLAGS contains threshold for predictions
	meta supplies labels and colors for drawing
	"""
	# meta
	threshold = FLAGS.threshold
	C, B, S = meta['classes'], meta['num'], meta['side']
	colors, labels = meta['colors'], meta['labels']

	boxes = []
	SS        =  S * S # number of grid cells
	prob_size = SS * C # class probabilities
	conf_size = SS * B # confidences for each grid cell
	probs = predictions[0 : prob_size]
	confs = predictions[prob_size : (prob_size + conf_size)]
	cords = predictions[(prob_size + conf_size) : ]
	probs = probs.reshape([SS, C])
	confs = confs.reshape([SS, B])
	cords = cords.reshape([SS, B, 4])

	for grid in range(SS):
		for b in range(B):
			new_box   = BoundBox(C)
			new_box.c =  confs[grid, b]
			new_box.x = (cords[grid, b, 0] + grid %  S) / S
			new_box.y = (cords[grid, b, 1] + grid // S) / S
			new_box.w =  cords[grid, b, 2] ** 2
			new_box.h =  cords[grid, b, 3] ** 2
			new_box.id = '{}-{}'.format(grid, b)
			for c in range(C):
				new_box.probs[c] = new_box.c * probs[grid, c]
			boxes.append(new_box)

	# non max suppress boxes
	for c in range(C):
		for i in range(len(boxes)): boxes[i].class_num = c
		boxes = sorted(boxes, cmp=prob_compare)
		for i in range(len(boxes)):
			boxi = boxes[i]
			if boxi.probs[c] == 0: continue
			for j in range(i + 1, len(boxes)):
				boxj = boxes[j]
				boxij = box_intersection(boxi, boxj)
				boxja = boxj.w * boxj.h
				apart = boxij / boxja
				if apart >= .5:
					if boxi.probs[c] > boxj.probs[c]:
						boxes[j].probs[c] = 0.
					else:
						boxes[i].probs[c] = 0.

	imgcv = cv2.imread(img_path)
	h, w, _ = imgcv.shape
	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = 'object' * int(C < 2)
		label += labels[max_indx] * int(C > 1)
		if (max_prob > threshold):
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300)
			cv2.rectangle(imgcv, 
				(left, top), (right, bot), 
				colors[max_indx], thick)
			mess = '{}:{:.3f}'.format(label, max_prob)
			cv2.putText(imgcv, mess, (left, top - 12), 
				0, 1e-3 * h, colors[max_indx],thick/5)

	img_name = os.path.join('results', img_path.split('/')[-1])
	cv2.imwrite(img_name, imgcv)
