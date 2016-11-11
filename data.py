"""
file: ./data.py
includes: shuffle()
shuffle will load the cPickle dump parsed.bin inside
"""

import cPickle as pickle
from yolo.train import *

off_bound_msg = 'Random scale/translate sends obj off bound'

def shuffle(train_path, parsed, batch, epoch, meta):
	with open(parsed, 'rb') as f: data = pickle.load(f)[0]
	size = len(data)
	print 'Dataset of {} instance(s)'.format(size)
	if batch > size: exit('Error: batch size is too big')
	batch_per_epoch = int(size / batch)
	total = epoch * batch_per_epoch
	yield total

	for i in range(epoch):
		print 'EPOCH {}'.format(i+1)
		# Shuffle data
		shuffle_idx = np.random.permutation(np.arange(size))
		for b in range(batch_per_epoch):
			start_idx = b * batch
			end_idx = (b+1) * batch

			datum = list()
			x_batch = list()
			offbound = False
			for j in range(start_idx,end_idx):
				real_idx = shuffle_idx[j]
				this = data[real_idx]
				img, tensors = yolo_batch(train_path, this, meta)
				if img is None: offbound = True; break
				x_batch += [img]
				if datum == list():	datum = tensors
				else: 
					for i in range(len(datum)):
						new_datum_i = [datum[i], tensors[i]]
						datum[i] = np.concatenate(new_datum_i)		
			
			if offbound: print off_bound_msg; continue
			x_batch = np.concatenate(x_batch, 0)
			yield (x_batch, datum)
