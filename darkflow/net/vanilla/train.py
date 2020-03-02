import tensorflow as tf

_LOSS_TYPE = ['sse','l2', 'smooth',
			  'sparse', 'l1', 'softmax',
			  'svm', 'fisher'] # Select Loss among list of losses

def loss(self, net_out):
	m = self.meta # since called in framework take self.meta from that cls
	loss_type = self.meta['type'] # take the loss type
	assert loss_type in _LOSS_TYPE, \
	'Loss type {} not implemented'.format(loss_type) # If loss_type not in list raise error

	out = net_out # network output  
	out_shape = out.get_shape() # shape of output
	out_dtype = out.dtype.base_dtype # datatype of output
	_truth = tf.placeholders(out_dtype, out_shape) # using both to assign target placeholder for groundtruth

	self.placeholders = dict({ # Truth placeholder for the framework
			'truth': _truth
		})

	diff = _truth - out #error
	#according to user choice calcuate among the range of losses
	if loss_type in ['sse','12']:
		loss = tf.nn.l2_loss(diff)

	elif loss_type == ['smooth']:
		small = tf.cast(diff < 1, tf.float32)
		large = 1. - small
		l1_loss = tf.nn.l1_loss(tf.multiply(diff, large))
		l2_loss = tf.nn.l2_loss(tf.multiply(diff, small))
		loss = l1_loss + l2_loss

	elif loss_type in ['sparse', 'l1']:
		loss = l1_loss(diff)

	elif loss_type == 'softmax':
		loss = tf.nn.softmax_cross_entropy_with_logits(logits, y)
		loss = tf.reduce_mean(loss)

	elif loss_type == 'svm':
		assert 'train_size' in m, \
		'Must specify'
		size = m['train_size']
		self.nu = tf.Variable(tf.ones([train_size, num_classes]))
