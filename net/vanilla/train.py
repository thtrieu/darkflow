

_LOSS_TYPE = ['sse','l2','l1','cross-entropy']

def parse(FLAGS, meta):
	pass

def batch(FLAGS, meta, chunk):
	return chunk[0], {'truth': chunk[1]}

def loss(net):
	m = net.meta
	loss_type = net.meta['type']
	assert loss_type in ['sse', 'l2']

	out = net.out
	out_shape = out.get_shape()
	out_dtype = out.dtype.base_dtype
	_truth = tf.placeholders(out_dtype, out_shape)

	placeholders = dict({
			'truth': _truth
		})

	if loss_type in ['sse','12']:
		loss = tf.nn.l2_loss(_truth - out)
	elif loss_type == 'l1':
