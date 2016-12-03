from simple import *
from convolution import local, convolutional
from baseop import HEADER, LINE

op_types = {
	'convolutional': convolutional,
	'connected': connected,
	'maxpool': maxpool,
	'leaky': leaky,
	'dropout': dropout,
	'flatten': flatten,
	'avgpool': avgpool,
	'softmax': softmax,
	'identity': identity,
	'crop': crop,
	'local': local,
	'select': select,
	'route': route,
	'reorg': reorg
}

def op_create(*args):
	layer_type = list(args)[0].type
	return op_types[layer_type](*args)