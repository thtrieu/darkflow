import numpy as np
import os

available = [
	'[convolutional]', '[connected]',
	'[maxpool]', '[dropout]', '[avgpool]',
	'[softmax]'
]

def parser(model):
	"""
	Read the .cfg file to extract layers into `layers`
	as well as model-specific parameters into `meta`
	"""
	def _parse(l, i = 1):
		return l.split('=')[i].strip()

	with open(model, 'rb') as f:
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
				if line == str(): continue
				try: #
					key = _parse(line, 0)
					val = _parse(line, 1)
					layer[key] = val
				except:
					pass

	meta = layer # last layer contains meta info
	meta['model'] = model
	meta['inp_size'] = [h, w, c]
	return layers, meta

def cfg_yielder(model, binary):
	"""
	yielding each layer information to initialize `layer`
	"""
	layers, meta = parser(model); yield meta;
	h, w, c = meta['inp_size']; l = w * h * c

	last_convo = None; size = None;
	name = model.split('/')[-1]
	name = name.split('.')[0]
	weightf = binary + '{}.weights'.format(name)

	# Start yielding
	flat = False # flag for 1st dense layer
	conv = '.conv.' in weightf
	for i, d in enumerate(layers):
		
		if conv and i > last_convo: break
		if d['type'] == '[convolutional]':

			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size / 2

			w_ = (w + 2 * padding - size)/stride + 1
			h_ = (h + 2 * padding - size)/stride + 1
			

			batch_norm = d.get('batch_normalize', 0) or conv
			yield ['convolutional', size, c, n, 
				   stride, padding, batch_norm]
			w, h, c = w_, h_, n
			l = w * h * c
			if 'activation' in d:
				if d['activation'] != 'linear':
					yield [d['activation']]
			
		if d['type'] == '[maxpool]':
			stride = d.get('stride', 1)
			size = d.get('size', stride)
			padding = d.get('padding', (size-1)/2)

			yield ['maxpool', size, stride, padding]

			w_ = (w + 2*padding)/d['stride'] 
			h_ = (h + 2*padding)/d['stride']
			w, h = w_, h_
			l = w * h * c

		if d['type'] == '[avgpool]':
			flat = True; l = c
			yield ['avgpool']

		if d['type'] == '[softmax]':
			yield ['softmax', d['groups']]

		if d['type'] == '[connected]':
			if not flat:
				yield ['flatten']
				flat = True
			yield ['connected', l, d['output']]
			l = d['output']
			if 'activation' in d:
				if d['activation'] != 'linear':
					yield [d['activation']]

		if d['type'] == '[dropout]': 
			yield ['dropout', d['probability']]

	#exit()