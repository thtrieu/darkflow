import numpy as np
import math
import cv2
import os
#from scipy.special import expit
from utils.box import BoundBox, box_intersection, prob_compare

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def postprocess(self, net_out, img_path):
	"""
	Takes net output, draw net_out, save to disk
	"""
	# meta
	meta, FLAGS = self.meta, self.FLAGS

	H, W, C = meta['out_size']
	threshold = meta['thresh']
	C, B = meta['classes'], meta['num']
	anchors = meta['anchors']
	segment = C + 5

	boxes = list()
	for row in range(H):
		for col in range(W):
			for b in range(B):
				new_box = BoundBox(C)

				box_pos = b * segment
				conf_pos = box_pos + 4
				prob_pos = conf_pos + 1

				new_box.c = expit(net_out[row, col, conf_pos])
				new_box.x = (col + expit(net_out[row, col, box_pos + 0])) / W
				new_box.y = (row + expit(net_out[row, col, box_pos + 1])) / H
				new_box.w = math.exp(net_out[row, col, box_pos + 2]) * anchors[2 * b]
				new_box.h = math.exp(net_out[row, col, box_pos + 3]) * anchors[2 * b + 1]
				new_box.w /= W; new_box.h /= H

				probs = net_out[row, col, prob_pos: (prob_pos + C)]
				new_box.probs = _softmax(probs) * new_box.c
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


	colors = meta['colors']
	labels = meta['labels']
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

	outfolder = os.path.join(FLAGS.test, 'out') 
	img_name = os.path.join(outfolder, img_path.split('/')[-1])
	cv2.imwrite(img_name, imgcv)