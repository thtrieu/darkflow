import numpy as np
import os

def parser(model):
	"""
	Read the .cfg file to extract layers into `layers`
	as well as model-specific parameters into `meta`
	"""
	def _parse(l, i = 1):
		return l.split('=')[i].strip()

	with open(model, 'rb') as f:
		lines = f.readlines()		
	
	meta = dict(); layers = list() # will contains layers' info
	h, w, c = [int()]*3; layer = dict()
	for line in lines:
		line = line.strip()
		if '[' in line:
			if layer != dict(): 
				if layer['type'] == '[net]': 
					h = layer['height']
					w = layer['width']
					c = layer['channels']
					meta['net'] = layer
				else:
					if layer['type'] == '[crop]':
						h = layer['crop_height']
						w = layer['crop_width']
					#assert layer['type'] in available, \
					#'Layer {} not implemented'.format(layer['type'])
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

	meta.update(layer) # last layer contains meta info
	meta['model'] = model
	meta['inp_size'] = [h, w, c]
	return layers, meta

def cfg_yielder(model, binary):
	"""
	yielding each layer information to initialize `layer`
	"""
	layers, meta = parser(model); yield meta;
	h, w, c = meta['inp_size']; l = w * h * c

	# Start yielding
	flat = False # flag for 1st dense layer
	conv = '.conv.' in model
	for i, d in enumerate(layers):
		#print d['type'], h, w, c, l

		if d['type'] == '[crop]':
			yield ['crop']

		elif d['type'] == '[local]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			activation = d.get('activation', 'logistic')
			w_ = (w - 1 - (1 - pad) * (size - 1)) / stride + 1
			h_ = (h - 1 - (1 - pad) * (size - 1)) / stride + 1

			yield ['local', size, c, n, stride, pad, w_, h_]
			if activation != 'linear': yield [activation]
		
			w, h, c = w_, h_, n
			l = w * h * c

		elif d['type'] == '[convolutional]':

			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size / 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			yield ['convolutional', size, c, n, 
				   stride, padding, batch_norm]
			if activation != 'linear': yield [activation]

			w_ = (w + 2 * padding - size)/stride + 1
			h_ = (h + 2 * padding - size)/stride + 1
			w, h, c = w_, h_, n
			l = w * h * c

		elif d['type'] == '[maxpool]':
			stride = d.get('stride', 1)
			size = d.get('size', stride)
			padding = d.get('padding', (size-1)/2)

			yield ['maxpool', size, stride, padding]

			w_ = (w + 2*padding)/d['stride'] 
			h_ = (h + 2*padding)/d['stride']
			w, h = w_, h_
			l = w * h * c

		elif d['type'] == '[avgpool]':
			flat = True; l = c
			yield ['avgpool']

		elif d['type'] == '[softmax]':
			yield ['softmax', d['groups']]

		elif d['type'] == '[connected]':
			if not flat:
				yield ['flatten']
				flat = True
			yield ['connected', l, d['output']]
			l = d['output']
			if 'activation' in d:
				if d['activation'] != 'linear':
					yield [d['activation']]

		elif d['type'] == '[dropout]': 
			yield ['dropout', d['probability']]

		else:
			exit('Layer {} not implemented'.format(d['type']))