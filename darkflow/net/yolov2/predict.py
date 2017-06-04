import numpy as np
import math
import cv2
import os
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from darkflow.utils.box import BoundBox
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
	e_x = np.exp(x - np.max(x))
	out = e_x / e_x.sum()
	return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()

	boxes=box_constructor(meta,net_out)
	return boxes

def complementary(b, r, g):
	return (255-b, 255-r, 255-g)

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	textBuff = "["
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			line = 	('{"label":"%s",'
					'"confidence":%.2f,'
					'"topleft":{"x":%d,"y":%d},'
					'"bottomright":{"x":%d,"y":%d}},\n') % \
					(mess, confidence, left, top, right, bot)
			textBuff += line
			continue

		# Bounding box
		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		# Measure label sizes
		wt, ht = cv2.getTextSize(mess, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, thick//2)[0]
		#wc, hc = cv2.getTextSize(str(format(confidence, '.3f')), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, thick//2)[0]
		# Draw label backgrounds
		cv2.rectangle(imgcv,
			(left, top), (left+wt, top-5-ht),
			colors[max_indx], cv2.FILLED)
		#cv2.rectangle(imgcv,
		# (right-wc, bot), (right, bot+hc),
		#	colors[max_indx], cv2.FILLED)
		# Draw labels 
		cv2.putText(imgcv, mess, (left, top-5),
			0, 1e-3 * h, complementary(*colors[max_indx]),thick//2)
		#cv2.putText(imgcv, str(format(confidence, '.3f')), (right-wc, bot+hc),
		#		0, 1e-3 * h, complementary(*colors[max_indx]),thick//2)


	if not save: return imgcv
	# Removing trailing comma+newline adding json list terminator.
	textBuff = textBuff[:-2] + "]"
	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textBuff)
		return

	cv2.imwrite(img_name, imgcv)
