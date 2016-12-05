import numpy as np
import os
import time
import tensorflow as tf

train_stats = (
	'Training statistics: \n'
	'\tLearning rate : {}\n'
	'\tBatch size    : {}\n'
	'\tEpoch number  : {}\n'
	'\tBackup every  : {}'
)

def train(self):
	batches = self.shuffle()
	model = self.meta['name']

	loss_mva = None; total = int() # total number of batches
	for i, packet in enumerate(batches):
		if i == 0: 
			total = packet;
			args = [self.FLAGS.lr, self.FLAGS.batch]
			args+= [self.FLAGS.epoch, self.FLAGS.save]
			self.say(train_stats.format(*args))
			continue

		x_batch, datum = packet

		if i == 1: \
		assert set(list(datum)) == set(list(self.placeholders)), \
		'Feed and placeholders of loss op mismatched'

		feed_pair = [(self.placeholders[k], datum[k]) for k in datum]
		feed_dict = {holder:val for (holder,val) in feed_pair}
		for k in self.feed: feed_dict[k] = self.feed[k]
		feed_dict[self.inp] = x_batch

		_, loss = self.sess.run([self.train_op, self.loss], feed_dict)

		if loss_mva is None: loss_mva = loss
		loss_mva = .9 * loss_mva + .1 * loss
		step_now = self.FLAGS.load + i
		args = [step_now, loss, loss_mva]
		self.say('step {} - loss {} - moving ave loss {}'.format(*args))
		if i % (self.FLAGS.save/self.FLAGS.batch) == 0 or i == total:
			ckpt = os.path.join(self.FLAGS.backup, '{}-{}'.format(model, step_now))
			self.say('Checkpoint at step {}'.format(step_now))
			self.saver.save(self.sess, ckpt)


def predict(self):
	inp_path = self.FLAGS.test
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
			expanded = np.expand_dims(this_inp, 0)
			inp_feed.append(expanded)
		all_inp = new_all

		feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
	
		self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
		start = time.time()
		out = self.sess.run(self.out, feed_dict)
		stop = time.time(); last = stop - start

		self.say('Total time = {}s / {} inps = {} ips'.format(
			last, len(inp_feed), len(inp_feed) / last))

		self.say('Post processing {} inputs ...'.format(len(inp_feed)))
		start = time.time()
		for i, prediction in enumerate(out):
			self.framework.postprocess(prediction,
				os.path.join(inp_path, all_inp[i]))
		stop = time.time(); last = stop - start
		self.say('Total time = {}s / {} inps = {} ips'.format(
			last, len(inp_feed), len(inp_feed) / last))
