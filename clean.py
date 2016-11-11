"""
file: ./clean.py
includes: a script to parse Pascal VOC data
this script produces the binary file parsed.bin, which contains
a cPickle dump of a list. Each element in the list corresponds
to an image, the element in turn contains a list of  parsed bounding 
boxes coordinates and asscociated classes of each object defined
in labels.txt. If labels.txt is left blank, the default choice of
all twenty objects are used (see list labels20 below).

The cPickle dump will be used mainly by ./data.py, inside function
shuffle(). shuffle() will shuffle and cut the dump into batches,
preprocess them so that they are ready to be fed into net.

WARNING: this script is messy, it hurts to read :(
"""

import os
import numpy as np
import cv2
import cPickle as pickle
import sys

if len(sys.argv) == 1:
	ANN = '../pascal/VOCdevkit/ANN'
else:
	ANN = sys.argv[1]

# ---- CONSTANTS-------
labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
	"bus", "car", "cat", "chair", "cow", "diningtable", "dog",
	"horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
	"train", "tvmonitor"]

pick = list()
with open('labels.txt', 'r') as f:
	pick = [l.strip() for l in f.readlines()]
if pick == list(): pick = labels20

def pp(l):
	for i in l: print '{}: {}'.format(i,l[i])

def parse(line):
	x = line.split('>')[1].split('<')[0]
	try:
		r = int(x)
	except:
		r = x
	return r

dumps = list()
tempdir = os.getcwd()
os.chdir(ANN)
size = len(os.listdir('.'))

for i, file in enumerate(os.listdir('.')):
	
	sys.stdout.write('\r')
	percentage = 1. * i / size
	progress = int(percentage * 20)
	sys.stdout.write('[{}>{}]{:.0f}%'.format(progress*'=',' '*(19-progress),percentage*100))
	sys.stdout.flush()
	
	if file.split('.')[1] != 'xml':
		continue
	with open(file, 'r') as f:
		lines = f.readlines()

	w = h = int()
	all = current = list()
	name = str()
	obj = False
	for i in range(len(lines)):
		line = lines[i]
		if '<width>' in line:
			w = parse(line)
		if '<height>' in line:
			h = parse(line)
		if '<object>' in line:
			obj = True
		if '</object>' in line:
			obj = False
		if '<part>' in line:
			obj = False
		if '</part>' in line:
			obj = True
		if not obj: continue
		if '<name>' in line:
			if current != list() and current[0] in pick:
					all += [current]
			current = list()
			name = parse(line)
			if name not in pick: 
				obj = False
				continue
			current = [name,None,None,None,None]
		if len(current) != 5: continue
		xn = '<xmin>' in line
		xx = '<xmax>' in line
		yn = '<ymin>' in line
		yx = '<ymax>' in line
		if xn: current[1] = parse(line)
		if xx: current[3] = parse(line)
		if yn: current[2] = parse(line)
		if yx: current[4] = parse(line)

	if current != list() and current[0] in pick:
		all += [current]

	if all == list(): continue
	jpg = file.split('.')[0]+'.jpg'
	add = [[jpg, [w, h, all]]]
	dumps += add


stat = dict()
for dump in dumps:
	all = dump[1][2]
	for current in all:
		if current[0] in pick:
			if current[0] in stat:
				stat[current[0]]+=1
			else:
				stat[current[0]] =1

print 
print 'Statistics:'
pp(stat)
print 'Dataset size: {}'.format(len(dumps))
with open('parsed.bin', 'wb') as f:
	pickle.dump([dumps],f,protocol=-1)
os.chdir(tempdir)