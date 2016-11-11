import numpy as np
import os

def _parse(l): return l.split('=')[1].strip()
    
def parser(model):
	"""
	Read the .cfg file to extract layers into `s`
	as well as model-specific parameters into `meta`
	"""
	with open('configs/yolo-{}.cfg'.format(model), 'rb') as f:
		lines = f.readlines()		
	
	s = [] # will contains layers' info
	add = dict()
	for line in lines:
		line = line.strip()
		# deepnet general layers
		if '[' in line:
			if add != {}: s += [add]
			add = {'type':line}
		else:
			try:
				i = float(_parse(line))
				if i == int(i): i = int(i)
				add[line.split('=')[0]] = i
			except:
				try:
					if _parse(line) == 'leaky':
						add['activation'] = 'leaky'
				except:
					pass
	add['model'] = model
	return s, add

def discoverer(weightf, s):
	"""
	discoverer returns:
	1. index of last convolutional layer
	2. the expected size of this conv layer's kernel
	"""
	allbytes = os.path.getsize(weightf)
	allfloat = allbytes/4; allfloat -= 4 
	last_convo = int() 
	for i, d in enumerate(s):
		if len(d) >= 4:
			last_convo = i
	channel = 3; dense = False # flag for 1st dense layer
	out = int() 
	for i, d in enumerate(s):
		# ignore darknet specifications
		if 'batch' in d: continue
		if 'crop_width' in d: continue
		if 'side' in d: continue
	
		if d['type'] == '[convolutional]': 
			kernel = d['size'] ** 2 * channel * d['filters']
			allfloat -= kernel + d['filters']
			channel = d['filters']
			if 'batch_normalize' in d:
				allfloat -= 2 * d['filters']
		elif d['type'] == '[connected]':
			if dense is False: 
				out = out1 = d['output'] 
				dense = True; continue 
			weight = out * d['output']
			allfloat -= weight + d['output']
			out = d['output']

	allfloat -= out1 # substract the bias
	if allfloat <= 0:
		message = 'yolo-{}.cfg suggests a bigger size'
		message += ' than yolo-{}.weights actually is'
		exit('Error: {}'.format(message.format(model, model)))
	
	# expected size of last convolution kernel
	size = (np.sqrt(1.*allfloat/out1/channel))
	print 'Last convolutional kernel size = {}'.format(size)
	size = int(size)
	n = last_convo + 1
	while 'output' not in s[n]:
		size *= s[n].get('size',1)
		n += 1
	return last_convo, size

def cfg_yielder(model, undiscovered = True):
	"""
	yielding each layer information, if model is discovered 
	for the first time (undiscovered = True), discoverer
	will be employed
	"""
	
	layers, meta = parser(model); yield meta

	if undiscovered:
		weightf = 'yolo-{}.weights'.format(model)
		last_convo, size = discoverer(weightf, layers)
	else: last_convo = None; size = None

	# Start yielding
	w = 448; h = 448; c = 3; l = w * h * c
	yield ['CROP']; flat = False # flag for 1st dense layer
	for i, d in enumerate(layers):
		# ignore darknet specifications
		if 'batch' in d: continue
		if 'crop_width' in d: continue
		if 'side' in d: continue

		if d['type'] == '[convolutional]':
			mult = (d['size'] == 3) 
			mult *= (d['stride'] != 2) + 1.
			if d['size'] == 1: d['pad'] = 0
			new = (w + mult * d['pad'] - d['size'])
			new /= d['stride']
			new = int(np.floor(new + 1.))
			if i == last_convo:
    			# signal tfnet to figure out the pad itself
				# to achieve the desired `size`. Namely, to
				# use the negative sign:
				d['pad'] = -size
				new = size
			yield ['conv', d['size'], c, d['filters'], 
				    h, w, d['stride'], d['pad']]	
			w = h = new
			c = d['filters']
			l = w * h * c
			if 'batch_normalize' in d: 
				yield['bnrm', 0, 0, c, 0, 0]
			if 'activation' in d: yield ['leaky']
			
		if d['type'] == '[maxpool]':
			yield ['pool', d['size'], 0, 
				0, 0, 0, d['stride'], 0]
			new = (w * 1.0 - d['size'])/d['stride'] + 1
			new = int(np.floor(new))
			w = h = new
			l = w * h * c

		if d['type'] == '[connected]':
			if not flat:
				yield ['flatten']
				flat = True
			yield ['conn'] + [0] * 5 + [l, d['output']]
			l = d['output']
			if 'activation' in d: yield ['leaky']

		if d['type'] == '[dropout]': 
			yield ['drop', d['probability']]