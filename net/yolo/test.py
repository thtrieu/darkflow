from utils.im_transform import imcv2_recolor, imcv2_affine_trans
from utils.box import BoundBox, box_intersection, prob_compare
import numpy as np
import cv2
import os

def preprocess(self, imPath, allobj = None):
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
		im = imcv2_recolor(im)

	h, w, c = self.meta['inp_size']
	imsz = cv2.resize(im, (h, w))
	imsz = imsz / 255.
	imsz = imsz[:,:,::-1]
	if allobj is None: return imsz
	return imsz #, np.array(im) # for unit testing
	

def postprocess(self, predictions, img_path):
	"""
	Takes net output, draw predictions, save to disk
	"""
	meta, FLAGS = self.meta, self.FLAGS
	threshold, sqrt = FLAGS.threshold, meta['sqrt'] + 1
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
			new_box.w =  cords[grid, b, 2] ** sqrt
			new_box.h =  cords[grid, b, 3] ** sqrt
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

	outfolder = os.path.join(FLAGS.test, 'out') 
	img_name = os.path.join(outfolder, img_path.split('/')[-1])
	cv2.imwrite(img_name, imgcv)
