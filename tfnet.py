import tensorflow as tf
import numpy as np
import os
import time
from drawer import *
from data import shuffle
from yolo import *
import subprocess
import sys

class SimpleNet(object):

	labels = list()
	colors = list()
	C = int()
	model = str()
	step = int()
	learning_rate = float()
	scale_prob = float()
	scale_conf = float()
	scale_noobj = float()
	scale_coor = float()
	save_every = int()

	def __init__(self, yolo, FLAGS):
		self.model = yolo.model
		self.S = yolo.S
		self.labels = yolo.labels
		self.C = len(self.labels)

		base = int(np.ceil(pow(self.C, 1./3)))
		for x in range(len(self.labels)):
			self.colors += [to_color(x, base)]		

		self.inp = tf.placeholder(tf.float32,
			[None, 448, 448, 3], name = 'input')
		self.drop = tf.placeholder(tf.float32, name = 'dropout')
		
		now = self.inp
		for i in range(yolo.layer_number):
			print now.get_shape()
			l = yolo.layers[i]
			if l.type == 'CONVOLUTIONAL':
				if l.pad < 0:
					size = np.int(now.get_shape()[1])
					expect = -(l.pad + 1) * l.stride # there you go bietche 
					expect += l.size - size
					padding = [expect / 2, expect - expect / 2]
					if padding[0] < 0: padding[0] = 0
					if padding[1] < 0: padding[1] = 0
				else:
					padding = [l.pad, l.pad]
				l.pad = 'VALID'
				now = tf.pad(now, [[0, 0], padding, padding, [0, 0]])
				if FLAGS.savepb:
					b = tf.constant(l.biases)
					w = tf.constant(l.weights)
				else:
					b = tf.Variable(l.biases)
					w = tf.Variable(l.weights)
				now = tf.nn.conv2d(now, w,
					strides=[1, l.stride, l.stride, 1],
					padding=l.pad)
				now = tf.nn.bias_add(now, b)
				now = tf.maximum(0.1 * now, now)			
			elif l.type == 'MAXPOOL':
				l.pad = 'VALID'
				now = tf.nn.max_pool(now, 
					padding = l.pad,
					ksize = [1,l.size,l.size,1], 
					strides = [1,l.stride,l.stride,1])			
			elif l.type == 'FLATTEN':
				now = tf.transpose(now, [0,3,1,2])
				now = tf.reshape(now, 
					[-1, int(np.prod(now.get_shape()[1:]))])			
			elif l.type == 'CONNECTED':
				name = str()
				if i == yolo.layer_number - 1: name = 'output'
				else: name = 'conn'
				if FLAGS.savepb:
					b = tf.constant(l.biases)
					w = tf.constant(l.weights)
				else:
					b = tf.Variable(l.biases)
					w = tf.Variable(l.weights)
				now = tf.nn.xw_plus_b(now, w, b, name = name)
			elif l.type == 'LEAKY':
				now = tf.maximum(0.1 * now, now)
			elif l.type == 'DROPOUT':
				if not FLAGS.savepb:
					print ('dropout')
					now = tf.nn.dropout(now, keep_prob = self.drop)
		print now.get_shape()
		self.out = now

	def setup_meta_ops(self, FLAGS):
		self.save_every = FLAGS.save
		self.learning_rate = FLAGS.lr
		scales = [float(f) for i, f in enumerate(FLAGS.scale.split(','))]
		self.scale_prob, self.scale_conf, self.scale_noobj, self.scale_coor = scales 
		if FLAGS.gpu > 0: 
			percentage = min(FLAGS.gpu, 1.)
			print 'gpu mode {} usage'.format(percentage)
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=percentage)
			self.sess = tf.Session(config = tf.ConfigProto(
				allow_soft_placement = True,
				log_device_placement = False,
				gpu_options = gpu_options))
		else:
			print 'cpu mode'
			self.sess = tf.Session(config = tf.ConfigProto(
				allow_soft_placement = False,
				log_device_placement = False))
		if FLAGS.train: self.decode()
		if FLAGS.savepb: 
			self.savepb('graph-{}.pb'.format(self.model))
			sys.exit()
		else: self.saver = tf.train.Saver(tf.all_variables(), max_to_keep = FLAGS.keep)
		self.sess.run(tf.initialize_all_variables())
		if FLAGS.load:
			load_point = 'backup/model-{}'.format(self.step)
			print 'loading from {}'.format(load_point)
			self.saver.restore(self.sess, load_point)

	def savepb(self, name):
		print 'Saving pb to {}'.format(name)
		tf.train.write_graph(self.sess.graph_def,'./', name, as_text = False)

	def to_constant(self, inc = 0):
		with open('binaries/yolo-{}-{}.weights'.format(
			self.model.split('-')[0], self.step + inc), 'w') as f:
			f.write(np.array([1]*4, dtype=np.int32).tobytes())
			for i, variable in enumerate(tf.trainable_variables()):
				val = variable.eval(self.sess)
				if len(val.shape) == 4:
					val = val.transpose([3,2,0,1])
				val = val.reshape([-1])
				f.write(val.tobytes())
	
	def decode(self):
			print ('Set up loss and train ops (may cause lag)...')
			SS = self.S * self.S
			self.true_class = tf.placeholder(tf.float32, #
				[None, SS * self.C])
			self.true_coo = tf.placeholder(tf.float32, #
				[None, SS * 2 * 4])
			self.class_idtf = tf.placeholder(tf.float32, #
				[None, SS * self.C])
			self.cooid1 = tf.placeholder(tf.float32, #
				[None, SS, 1, 4])
			self.cooid2 = tf.placeholder(tf.float32, #
				[None, SS, 1, 4])
			self.confs1 = tf.placeholder(tf.float32, #
				[None, SS])
			self.confs2 = tf.placeholder(tf.float32, #
				[None, SS])
			self.conid1 = tf.placeholder(tf.float32, #
				[None, SS])
			self.conid2 = tf.placeholder(tf.float32, #
				[None, SS])
			self.upleft = tf.placeholder(tf.float32, #
				[None, SS, 2, 2])
			self.botright = tf.placeholder(tf.float32, #
				[None, SS, 2, 2])

			coords = self.out[:, SS * (self.C + 2):]
			coords = tf.reshape(coords, [-1, SS, 2, 4])

			wh = tf.pow(coords[:,:,:,2:4], 2) * 3.5;
			xy = coords[:,:,:,0:2]
			floor = xy - wh
			ceil = xy + wh

			# [batch, 49, box, xy]
			intersect_upleft = tf.maximum(floor, self.upleft)
			intersect_botright = tf.minimum(ceil, self.botright)
			intersect_wh = intersect_botright - intersect_upleft
			intersect_wh = tf.maximum(intersect_wh, 0.0)
			
			# [batch, 49, box]
			intersect_area1 = tf.mul(intersect_wh[:,:,0,0], intersect_wh[:,:,0,1])
			intersect_area2 = tf.mul(intersect_wh[:,:,1,0], intersect_wh[:,:,1,1])
			inferior_cell = intersect_area1 > intersect_area2
			inferior_cell = tf.to_float(inferior_cell)

			# [batch, 49]
			confs1 = tf.mul(inferior_cell, self.confs1) 
			confs2 = tf.mul((1.-inferior_cell), self.confs2)
			confs1 = tf.expand_dims(confs1, -1)
			confs2 = tf.expand_dims(confs2, -1)
			confs = tf.concat(2, [confs1, confs2])
			# [batch, 49, 2]

			mult = inferior_cell
			conid1 =  tf.mul(mult, self.conid1)
			conid2 =  tf.mul((1. - mult), self.conid2)
			conid1 = tf.expand_dims(conid1, -1)
			conid2 = tf.expand_dims(conid2, -1)
			conid = tf.concat(2, [conid1, conid2])
			# [batch, 49, 2]

			times = tf.expand_dims(inferior_cell, -1) # [batch, 49, 1]
			times = tf.expand_dims(times, 2) # [batch, 49, 1, 1]
			times = tf.concat(3, [times]*4) # [batch, 49, 1, 4]
			cooid1 = tf.mul(times, self.cooid1)
			cooid2 = (1. - times) * self.cooid2
			cooid = tf.concat(2, [cooid1, cooid2]) # [batch, 49, 2, 4]

			confs = tf.reshape(confs,
				[-1, int(np.prod(confs.get_shape()[1:]))])
			conid = tf.reshape(conid,
				[-1, int(np.prod(conid.get_shape()[1:]))])
			cooid = tf.reshape(cooid,
				[-1, int(np.prod(cooid.get_shape()[1:]))])

			conid = conid + tf.to_float(conid > .5) * (self.scale_conf - 1.)
			conid = conid + tf.to_float(conid < .5) * self.scale_noobj

			true = tf.concat(1,[self.true_class, confs, self.true_coo])
			idtf = tf.concat(1,[self.class_idtf * self.scale_prob, conid,
								cooid * self.scale_coor])

			self.loss = tf.pow(self.out - true, 2)
			self.loss = tf.mul(self.loss, idtf)
			self.loss = tf.reduce_sum(self.loss, 1)
			self.loss = .5 * tf.reduce_mean(self.loss)

			optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
			gradients = optimizer.compute_gradients(self.loss)
			self.train_op = optimizer.apply_gradients(gradients)

	def train(self, train_set, annotate, batch_size, epoch):
		batches = shuffle(train_set, annotate, self.C, self.S, batch_size, epoch)
		for i, batch in enumerate(batches):
			x_batch, datum = batch
			feed_dict = {
				self.inp : x_batch,
				self.drop : .5,
				self.true_class : datum[0],
				self.confs1 : datum[1],
				self.confs2 : datum[2],
				self.true_coo : datum[3],
				self.upleft : datum[4],
				self.botright : datum[5],
				self.class_idtf : datum[6],
				self.conid1 : datum[7],
				self.conid2 : datum[8],
				self.cooid1 : datum[9],
				self.cooid2 : datum[10],
			}
			_, loss = self.sess.run([self.train_op, self.loss], feed_dict)
			print 'step {} - batch {} - loss {}'.format(1+i+self.step, 1+i, loss)
			if (i+1) % (self.save_every/batch_size) == 0:
				print 'save checkpoint and binaries at step {}'.format(self.step+i+1)
				self.saver.save(self.sess, 'backup/model-{}'.format(self.step+i+1))
				self.to_constant(inc = i+1)

		print 'save checkpoint and binaries at step {}'.format(self.step+i+1)
		self.saver.save(self.sess, 'backup/model-{}'.format(self.step+i+1))
		self.to_constant(inc = i+1)

	def predict(self, FLAGS):
		img_path = FLAGS.test
		threshold = FLAGS.threshold
		all_img_ = os.listdir(img_path)
		batch = min(FLAGS.batch, len(all_img_))
		for j in range(len(all_img_)/batch):
			img_feed = list()
			all_img = all_img_[j*batch: (j*batch+batch)]
			new_all = list()
			for img in all_img:
				if '.xml' in img: continue
				new_all += [img]
				this_img = '{}/{}'.format(img_path, img)
				this_img = crop(this_img)
				img_feed.append(this_img)
				img_feed.append(this_img[:,:,::-1,:])
			all_img = new_all

			feed_dict = {
				self.inp : np.concatenate(img_feed, 0), 
				self.drop : 1.0
			}
		
			print ('Forwarding {} images ...'.format(len(img_feed)))
			start = time.time()
			out = self.sess.run([self.out], feed_dict)
			stop = time.time()
			last = stop - start
			print ('Total time = {}s / {} imgs = {} fps'.format(
				last, len(img_feed), len(img_feed) / last))
			for i, prediction in enumerate(out[0]):
				draw_predictions(
					prediction,
					'{}/{}'.format(img_path, all_img[i/2]), 
					i % 2, threshold,
					self.C, self.S, self.labels, self.colors)
			print ('Results stored in results/')
