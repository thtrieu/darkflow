import numpy as np		
import cv2

def im_np_recolor(im):
	# `im` is a numpy python object
	# recolor `im` by adding in random
	# intensity transformations, DO NOT
	# perform shift/scale or rotate here
	# ADD YOUR CODE BELOW:
	alpha = np.random.uniform() + .5
	beta = np.random.uniform() * 40 - 20
	im = im * alpha
	im = im + beta
	return im

def imcv2_affine_trans(im):
	"""
	transform cv2 image
	with random translation, scale,
	illumination, scale
	"""
	# Scale and translate
	h, w, c = im.shape
	scale = np.random.uniform() / 10. + 1.
	max_offx = (scale-1.) * w
	max_offy = (scale-1.) * h
	offx = int(np.random.uniform() * max_offx)
	offy = int(np.random.uniform() * max_offy)
	im = cv2.resize(im, (0,0), fx = scale, fy = scale)
	im = im[offy : (offy + h), offx : (offx + w)]
	flip = np.random.binomial(1, .5)
	if flip: im = cv2.flip(im, 1)
	return im, [w, h, c], [scale, [offx, offy], flip]