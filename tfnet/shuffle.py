"""
file: ./data.py
includes: shuffle()
"""
import numpy as np

off_bound_msg = 'Random scale/translate sends obj off bound'

def shuffle(FLAGS, meta, framework):
	"""
	Call the specific to parse annotations, where or not doing the parse
	is up to the model. Then use the parsed object to yield minibatches
	minibatches will be preprocessed before yielding to be appropriate
	placeholders for model's loss evaluation.
	"""
	data = framework.parse(FLAGS, meta)
	size = len(data); batch = FLAGS.batch

	print 'Dataset of {} instance(s)'.format(size)
	if batch > size: exit('Error: batch size is too big')
	batch_per_epoch = int(size / batch)
	total = FLAGS.epoch * batch_per_epoch
	yield total

	for i in range(FLAGS.epoch):
		print 'EPOCH {}'.format(i+1)
		shuffle_idx = np.random.permutation(np.arange(size))
		for b in range(batch_per_epoch):
			start_idx = b * batch
			end_idx = (b+1) * batch

			datum = dict()
			x_batch = list()
			offbound = False
			for j in range(start_idx,end_idx):
				real_idx = shuffle_idx[j]
				this = data[real_idx]
				inp, tensors = framework.batch(FLAGS, meta, this)
				if inp is None: offbound = True; break
				x_batch += [inp]
				for k in tensors:
					if k not in datum: datum[k] = [tensors[k]]; continue
					datum[k] = np.concatenate([datum[k], [tensors[k]]])		
			
			if offbound: print off_bound_msg; continue
			x_batch = np.concatenate(x_batch, 0)
			yield (x_batch, datum)