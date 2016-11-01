from configs.process import cfg_yielder
from yolo import *
import numpy as np
import os
import sys


src = sys.argv[1]
try:
	des = sys.argv[2]
except:
	des = src
	src = str()

wlayer = ['CONVOLUTIONAL', 'CONNECTED']
class collector(object):
	def __init__(self, yolo):
		self.i = 0
		self.yolo = yolo
	def inc(self):
		while self.yolo.layers[self.i].type not in wlayer:
			self.i += 1
			if self.i == len(self.yolo.layers):
				break
	def give(self):
		self.inc()
		l = self.yolo.layers[self.i]
		w = l.weights
		if l.type == 'CONVOLUTIONAL':
			w = w.transpose([3,2,0,1])
		w = w.reshape([-1])
		w = np.concatenate((l.biases, w))
		self.i += 1
		return np.float32(w)


mark = int(1)
writer = open('yolo-{}.weights'.format(des),'w')
writer.write(np.int32([int(0)]*4).tobytes())
offset = int(16)

if src != str():
	yolo = YOLO(src)
	col = collector(yolo)
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
	if mark == i:
    		print 'none'
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
				scale = .05,
				size = (k[1]*k[1]*k[2]*k[3]+k[3],))
		else:
			w = np.random.normal(
				scale = .05,
				size = (k[6]*k[7]+k[7],))
		w = np.float32(w)
		writer.write(w.tobytes())
		offset += w.shape[0] * 4
writer.close()
print 'total size: {} bytes'.format(offset)