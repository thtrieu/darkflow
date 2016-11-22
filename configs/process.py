import numpy as np
import os

available = [
	'[convolutional]', '[connected]',
	'[maxpool]', '[dropout]'
]

def parser(config, model):
	"""
	Read the .cfg file to extract layers into `layers`
	as well as model-specific parameters into `meta`
	"""
	def _parse(l): return l.split('=')[1].strip()
	with open('{}{}.cfg'.format(config, model), 'rb') as f:
		lines = f.readlines()		
	
	layers = [] # will contains layers' info
	h, w, c = [int()]*3; layer = dict()
	for line in lines:
		line = line.strip()
		if '[' in line:
			if layer != dict(): 
				if layer['type'] == '[net]': 
					h = layer['height']
					w = layer['width']
					c = layer['channels']
				elif layer['type'] == '[crop]':
					h = layer['crop_height']
					w = layer['crop_width']
				else: 
					assert layer['type'] in available, \
					'Layer {} not implemented'.format(layer['type'])
					layers += [layer]				
			layer = {'type':line}
		else:
			try:
				i = float(_parse(line))
				if i == int(i): i = int(i)
				layer[line.split('=')[0]] = i
			except:
				try:
					if _parse(line) == 'leaky':
						layer['activation'] = 'leaky'
				except:
					pass
	meta = layer # last layer contains meta info
	meta['model'] = model
	meta['inp_size'] = [h, w, c]
	return layers, meta

def discoverer(weightf, s, c):
	"""
	discoverer returns:
	1. index of last convolutional layer
	2. the expected size of this conv layer's kernel
	"""
	allbytes = os.path.getsize(weightf)
	allfloat = allbytes/4.; allfloat -= 4 
	if allfloat != int(allfloat):
		msg = '{} might be corrupted, '
		msg += 'there is not an integer '
		msg += 'number of floats in there.'
		exit('Error:{}'.format(msg.format(weightf)))

	last_convo = int() 
	for i, d in enumerate(s):
		if d['type'] == '[convolutional]': 
			last_convo = i
	
	out = int() # output_dim of 1st dense layer
	channel = c; dense = False # for 1st dense layer
	for i, d in enumerate(s):
		if d['type'] == '[convolutional]': 
			kernel = d['size'] ** 2 * channel * d['filters']
			allfloat -= kernel + d['filters']
			channel = d['filters']
			if 'batch_normalize' in d: # scale, mean, var
				allfloat -= 3 * d['filters'] 
		elif d['type'] == '[connected]':
			if dense is False: 
				out = out1 = d['output'] 
				dense = True; continue 
			weight = out * d['output']
			allfloat -= weight + d['output']
			out = d['output']

	allfloat -= out1 # substract the bias
	if allfloat <= 0:
		msg = 'Configuration suggests a bigger size'
		msg += ' than {} actually is.'
		exit('Error: {}'.format(msg.format(weightf)))

	# expected size of last convolution kernel
	size = (np.sqrt(1.*allfloat/out1/channel))
	n = last_convo + 1
	while 'output' not in s[n]: 
		size *= s[n].get('size',1); n += 1
	print 'Last convolutional kernel size = {}'.format(size)
	return last_convo, int(size)

def cfg_yielder(model, binary, config):
	"""
	yielding each layer information, if model is discovered 
	for the first time (undiscovered = True), discoverer
	will be employed
	"""
	layers, meta = parser(config, model); yield meta;
	h, w, c = meta['inp_size']; l = w * h * c

	last_convo = None; size = None;
	weightf = binary + '{}.weights'.format(model)
	if os.path.isfile(weightf): # there is an assisting binary
		last_convo, size = discoverer(weightf, layers, c)

	# Start yielding
	flat = False # flag for 1st dense layer
	for i, d in enumerate(layers):

		if d['type'] == '[convolutional]':
			mult = (d['size'] == 3) 
			mult *= (d['stride'] != 2) + 1.
			if d['size'] == 1: d['pad'] = 0

			w_ = (w + mult * d['pad'] - d['size'])/d['stride']
			w_ = int(np.floor(w_ + 1.))
			h_ = (h + mult * d['pad'] - d['size'])/d['stride']
			h_ = int(np.floor(h_ + 1.))

			if i == last_convo:
    			# signal tfnet to figure out the pad itself
				# for achieving the desired `size`. Namely, 
				# to use the negative sign:
				d['pad'] = -size
				w_ = h_ = size

			batch_norm = d.get('batch_normalize', 0)
			yield ['convolutional', d['size'], c, d['filters'], 
				   d['stride'], d['pad'], batch_norm]
			w, h = w_, h_
			c = d['filters']
			l = w * h * c
			if 'activation' in d: yield ['leaky']
			
		if d['type'] == '[maxpool]':
			pad = d.get('pad', 0)
			yield ['maxpool', d['size'], d['stride'], pad]
			w_ = (w * 1.0 - d['size'])/d['stride'] + 1
			w_ = int(np.floor(w_))
			h_ = (h * 1.0 - d['size'])/d['stride'] + 1
			h_ = int(np.floor(h_))
			w, h = w_, h_
			l = w * h * c

		if d['type'] == '[connected]':
			if not flat:
				yield ['flatten']
				flat = True
			yield ['connected', l, d['output']]
			l = d['output']
			if 'activation' in d: yield ['leaky']

		if d['type'] == '[dropout]': 
			yield ['dropout', d['probability']]