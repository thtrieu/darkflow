"""
tfnet methods that involve flowing the graph
"""

from tfnet_help import shuffle
import numpy as np
import os
import time

def tf_train(self):
	batches = shuffle(self)

	print 'Training statistics:'
	print '\tLearning rate : {}'.format(self.FLAGS.lr)
	print '\tBatch size    : {}'.format(self.FLAGS.batch)
	print '\tEpoch number  : {}'.format(self.FLAGS.epoch)
	print '\tBackup every  : {}'.format(self.FLAGS.save)

	losses = list(); total = int() # total number of batches
	for i, packet in enumerate(batches):
		if i == 0: total = packet; continue
		x_batch, datum = packet

		if i == 1: \
		assert set(list(datum)) == set(list(self.placeholders)), \
		'Mismatch between placeholders and datum for loss evaluation'

		feed_pair = [(self.placeholders[k], datum[k]) for k in datum]
		feed_dict = {holder:val for (holder,val) in feed_pair}
		for k in self.feed: feed_dict[k] = self.feed[k]['feed']
		feed_dict[self.inp] = x_batch

		_, loss = self.sess.run([self.train_op, self.loss], feed_dict)

		losses += [loss]; step_now = self.FLAGS.load + i
		print 'step {} - loss {}'.format(step_now, loss)
		if i % (self.FLAGS.save/self.FLAGS.batch) == 0 or i == total:
			ckpt = os.path.join('backup', '{}-{}'.format(self.meta['model'], step_now))
			print 'Checkpoint at step {}'.format(step_now)
			self.saver.save(self.sess, ckpt)


def tf_predict(self):
	inp_path = self.FLAGS.testset
	all_inp_ = os.listdir(inp_path)
	all_inp_ = [i for i in all_inp_ if self.framework.is_inp(i)]
	if not all_inp_:
		msg = 'Failed to find any test files in {} .'
		exit('Error: {}'.format(msg.format(inp_path)))

	batch = min(self.FLAGS.batch, len(all_inp_))

	for j in range(len(all_inp_)/batch):
		inp_feed = list(); new_all = list()
		all_inp = all_inp_[j*batch: (j*batch+batch)]
		for inp in all_inp:
			new_all += [inp]
			this_inp = os.path.join(inp_path, inp)
			this_inp = self.framework.preprocess(this_inp)
			inp_feed.append(this_inp)
		all_inp = new_all

		feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
		for k in self.feed: feed_dict[k] = self.feed[k]['dfault']
	
		print ('Forwarding {} inputs ...'.format(len(inp_feed)))
		start = time.time()
		out = self.sess.run([self.out], feed_dict)
		stop = time.time(); last = stop - start
		print ('Total time = {}s / {} inps = {} ips'.format(
			last, len(inp_feed), len(inp_feed) / last))

		for i, prediction in enumerate(out[0]):
			self.framework.postprocess(prediction,
				os.path.join(inp_path, all_inp[i]),
				self.FLAGS, self.meta)
