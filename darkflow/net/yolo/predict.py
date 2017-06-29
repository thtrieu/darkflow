from ...utils.im_transform import imcv2_recolor, imcv2_affine_trans
from ...utils.box import BoundBox, box_iou, prob_compare
import numpy as np
import cv2
import os
import json
from ...cython_utils.cy_yolo_findboxes import yolo_box_constructor

def _fix(obj, dims, scale, offs):
	for i in range(1, 5):
		dim = dims[(i + 1) % 2]
		off = offs[(i + 1) % 2]
		obj[i] = int(obj[i] * scale - off)
		obj[i] = max(min(obj[i], dim), 0)

def resize_input(self, im):
	h, w, c = self.meta['inp_size']
	imsz = cv2.resize(im, (w, h))
	imsz = imsz / 255.
	imsz = imsz[:,:,::-1]
	return imsz

def process_box(self, b, h, w, threshold):
	max_indx = np.argmax(b.probs)
	max_prob = b.probs[max_indx]
	label = self.meta['labels'][max_indx]
	if max_prob > threshold:
		left  = int ((b.x - b.w/2.) * w)
		right = int ((b.x + b.w/2.) * w)
		top   = int ((b.y - b.h/2.) * h)
		bot   = int ((b.y + b.h/2.) * h)
		if left  < 0    :  left = 0
		if right > w - 1: right = w - 1
		if top   < 0    :   top = 0
		if bot   > h - 1:   bot = h - 1
		mess = '{}'.format(label)
		return (left, right, top, bot, mess, max_indx, max_prob)
	return None

def findboxes(self, net_out):
	meta, FLAGS = self.meta, self.FLAGS
	threshold = FLAGS.threshold
	
	boxes = []
	boxes = yolo_box_constructor(meta, net_out, threshold)
	
	return boxes

def preprocess(self, im, allobj = None):
	"""
	Takes an image, return it as a numpy tensor that is readily
	to be fed into tfnet. If there is an accompanied annotation (allobj),
	meaning this preprocessing is serving the train process, then this
	image will be transformed with random noise to augment training data,
	using scale, translation, flipping and recolor. The accompanied
	parsed annotation (allobj) will also be modified accordingly.
	"""
	if type(im) is not np.ndarray:
		im = cv2.imread(im)

	if allobj is not None: # in training mode
		result = imcv2_affine_trans(im)
		im, dims, trans_param = result
		scale, offs, flip = trans_param
		for obj in allobj:
			_fix(obj, dims, scale, offs)
			if not flip: continue
			obj_1_ =  obj[1]
			obj[1] = dims[0] - obj[3]
			obj[3] = dims[0] - obj_1_
		im = imcv2_recolor(im)

	im = self.resize_input(im)
	if allobj is None: return im
	return im#, np.array(im) # for unit testing

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw predictions, save to disk
	"""
	meta, FLAGS = self.meta, self.FLAGS
	threshold = FLAGS.threshold
	colors, labels = meta['colors'], meta['labels']

	boxes = self.findboxes(net_out)

	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im

	h, w, _ = imgcv.shape
	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			self.meta['colors'][max_indx], thick)
		cv2.putText(
			imgcv, mess, (left, top - 12),
			0, 1e-3 * h, self.meta['colors'][max_indx],
			thick // 3)


	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return	

	cv2.imwrite(img_name, imgcv)
