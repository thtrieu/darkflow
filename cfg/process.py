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
			layer = {'type': line}
		else:
			try:
				i = float(_parse(line))
				if i == int(i): i = int(i)
				layer[line.split('=')[0]] = i
			except:
				try:
					key = _parse(line, 0)
					val = _parse(line, 1)
					layer[key] = val
				except:
					# empty lines
					# crazy lines
					pass

	meta.update(layer) # last layer contains meta info
	meta['model'] = model # path to cfg, not model name
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
		#-----------------------------------------------------
		if d['type'] == '[crop]':
			yield ['crop', i]
		#-----------------------------------------------------
		elif d['type'] == '[local]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			activation = d.get('activation', 'logistic')
			w_ = (w - 1 - (1 - pad) * (size - 1)) / stride + 1
			h_ = (h - 1 - (1 - pad) * (size - 1)) / stride + 1
			yield ['local', i, size, c, n, stride, 
					pad, w_, h_, activation]
			if activation != 'linear': yield [activation, i]
			w, h, c = w_, h_, n
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[convolutional]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size / 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			yield ['convolutional', i, size, c, n, 
				   stride, padding, batch_norm,
				   activation]
			if activation != 'linear': yield [activation, i]
			w_ = (w + 2 * padding - size)/stride + 1
			h_ = (h + 2 * padding - size)/stride + 1
			w, h, c = w_, h_, n
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[maxpool]':
			stride = d.get('stride', 1)
			size = d.get('size', stride)
			padding = d.get('padding', (size-1)/2)

			yield ['maxpool', i, size, stride, padding]

			w_ = (w + 2*padding)/d['stride'] 
			h_ = (h + 2*padding)/d['stride']
			w, h = w_, h_
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[avgpool]':
			flat = True; l = c
			yield ['avgpool', i]
		#-----------------------------------------------------
		elif d['type'] == '[softmax]':
			yield ['softmax', i, d['groups']]
		#-----------------------------------------------------
		elif d['type'] == '[connected]':
			if not flat:
				yield ['flatten', i]
				flat = True
			activation = d.get('activation', 'logistic')
			yield ['connected', i, l, d['output'], activation]
			if activation != 'linear': yield [activation, i]
			l = d['output']
		#-----------------------------------------------------
		elif d['type'] == '[dropout]': 
			yield ['dropout', i, d['probability']]
		#-----------------------------------------------------
		elif d['type'] == '[select]':
			if not flat:
				yield ['flatten', i]
				flat = True
			activation = d.get('activation', 'logistic')
			d['keep'] = d['keep'].split('/')
			classes = int(d['keep'][-1])
			keep = [int(c) for c in d['keep'][0].split(',')]
			keep_n = len(keep)
			train_from = classes * d['bins']
			for count in range(d['bins']-1):
				for num in keep[-keep_n:]:
					keep += [num + classes]
			yield ['select', i, l, d['old_output'],
				   d['output'], keep, train_from,
				   activation]
			if activation != 'linear': yield [activation, i]
			l = d['output']
		#-----------------------------------------------------
		elif d['type'] == '[route]': # add new layer here
			routes = d['layers']
			if type(routes) is str:
				routes = [int(x.strip()) for x in routes.split(',')]
			else: routes = [routes]
			routes = [i + x for x in routes if x < 0]
			for j, x in enumerate(routes):
				lx = layers[x]; xtype = lx['type']
				_size = lx['_size'][:3]
				if not j: w, h, c = _size
				else: # simply for checking
					w_, h_, c_ = _size
					assert w_ == w and h_ == h, \
					'Routing incompatible conv sizes'
					c += c_
			yield ['route', i, routes]
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[reorg]':
			stride = d.get('stride', 1)
			yield ['reorg', i, stride]
			w /= stride; h /=stride; c *= stride ** 2
			l = w * h * c
		#-----------------------------------------------------
		else:
			exit('Layer {} not implemented'.format(d['type']))

		d['_size'] = list([h, w, c, l, flat])

	if not flat: meta['out_size'] = [h, w, c]
	else: meta['out_size'] = l