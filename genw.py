from configs.process import *
from yolo.train import *
from tensorflow import flags
from darknet import *
import numpy as np
import os
import sys

flags.DEFINE_string("src", "", "source of recollection: model name if source is complete, file name if source is partial, blank if no source")
flags.DEFINE_string("des", "", "name of new model")
flags.DEFINE_float("std", 1e-2, "standard deviation of random initialization")
FLAGS = flags.FLAGS
src = FLAGS.src
des = FLAGS.des

wlayer = ['convolutional', 'connected']
class collector(object):
	def __init__(self, net):
		self.i = 0
		self.net = net
	def inc(self):
		while self.net.layers[self.i].type not in wlayer:
			self.i += 1
			if self.i == len(self.net.layers):
				break
	def give(self):
		self.inc()
		l = self.net.layers[self.i]
		w = list()
		if l.type == 'convolutional':
			w += [l.p['biases']]
			if l.batch_norm:
				w += [l.p['scale']]
				w += [l.p['mean']]
				w += [l.p['var']]
			kernel = l.p['kernel']
			kernel = kernel.transpose([3,2,0,1])
			kernel = kernel.reshape([-1])
			w += [kernel]
		if l.type == 'connected':
			w += [l.p['biases']]
			w += [l.p['weights'].reshape([-1])]
		w = np.concatenate(w)
		self.i += 1
		return np.float32(w)


mark = int(1)
writer = open('yolo-{}.weights'.format(des),'w')
writer.write(np.int32([int(0)]*4).tobytes())
offset = int(16)

if src != str():
	partial = False
	if ".weights" in src:
		partial = True
		src = des # same structure
	net = Darknet(src, partial)
	col = collector(net)
	flag = True

	# PHASE 01: recollect
	print 'Recollect:'
	for i, k in enumerate(zip(cfg_yielder(des, undiscovered = False), 
							cfg_yielder(src, undiscovered = False))):
		if not i: continue
		if k[0][:] != k[1][:] and k[0][0] in ['conv', 'conn']:
			flag = False
		if flag:
			k = k[0]
			if k[0] not in ['conv', 'conn']: continue
			w = col.give()
			writer.write(w.tobytes())
			offset += w.shape[0] * 4
			print k
		elif not flag:
			mark = i
			break
else:
    flag = False

# PHASE 02: random init
print 'Random init:'
if not flag:
	for i, k in enumerate(cfg_yielder(des, undiscovered = False)):
		if i < mark: continue
		if k[0] not in ['conv','conn']: continue
		print k
		if k[0] == 'conv':
			w = np.random.normal(
				scale = FLAGS.std,
				size = (k[1]*k[1]*k[2]*k[3]+k[3],))
		else:
			w = np.random.normal(
				scale = FLAGS.std,
				size = (k[6]*k[7]+k[7],))
		w = np.float32(w)
		writer.write(w.tobytes())
		offset += w.shape[0] * 4
writer.close()
print 'total size: {} bytes'.format(offset)