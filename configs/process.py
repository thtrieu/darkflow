import numpy as np
import os

def cfg_yielder(model, mode = True):
	# Parse ---------------------------------------
	with open('configs/yolo-{}.cfg'.format(model), 'rb') as f:
		lines = f.readlines()

	s = []
	S = int()
	add = dict()
	for line in lines:
		line = line.strip()
		if 'side' in line:
			S = int(line.split('=')[1].strip())
		if '[' in line:
			if add != {}:
				s += [add]
			add = dict()
		else:
			try:
				i = float(line.split('=')[1].strip())
				if i == int(i): i = int(i)
				add[line.split('=')[0]] = i
			except:
				try:
					if line.split('=')[1] == 'leaky' and 'output' in add:
						add[line.split('=')[0]] = line.split('=')[1]
				except:
					pass
	yield S
	# Interprete---------------------------------------
	weightf = 'yolo-{}.weights'.format(model)
	if mode:
		allbytes = os.path.getsize('yolo-{}.weights'.format(model))
		allbytes /= 4
		allbytes -= 4
		last_convo = int()
		for i, d in enumerate(s):
			if len(d) == 4:
				last_convo = i
		flag = False
		channel = 3
		out = int()
		for i, d in enumerate(s):
			if len(d) == 4:
				allbytes -= d['size'] ** 2 * channel * d['filters']
				allbytes -= d['filters']
				channel = d['filters']
			elif 'output' in d:
				if flag is False:
					out = out1 = d['output']
					flag = True
					continue
				allbytes -= out * d['output']
				allbytes -= d['output']
				out = d['output']
		allbytes -= out1
		size = (np.sqrt(allbytes/out1/channel))
		size = int(size)
		n = last_convo + 1
		while 'output' not in s[n]:
			size *= s[n].get('size',1)
			n += 1
	else:
		last_convo = None
		size = None

	w = 448
	h = 448
	c = 3
	l = w * h * c
	flat = False
	yield ['CROP']
	for i, d in enumerate(s):
		#print w, h, c, l
		flag = False
		if len(d) == 4:
			mult = (d['size'] == 3) 
			mult *= (d['stride'] != 2) + 1.
			if d['size'] == 1: d['pad'] = 0
			new = (w + mult * d['pad'] - d['size'])
			new /= d['stride']
			new = int(np.floor(new + 1.))
			if i == last_convo:
				d['pad'] = -size
				new = size
			yield ['conv', d['size'], c, d['filters'], 
				    h, w, d['stride'], d['pad']]	
			w = h = new
			c = d['filters']
			l = w * h * c
			#print w, h, c
		if len(d) == 2:
			if 'output' not in d:
				yield ['pool', d['size'], 0, 
					0, 0, 0, d['stride'], 0]
				new = (w * 1.0 - d['size'])/d['stride'] + 1
				new = int(np.floor(new))
				w = h = new
				l = w * h * c
			else:
				if not flat:
					flat = True
					yield ['FLATTEN']
				yield ['conn', 0, 0,
				0, 0, 0, l, d['output']]
				l = d['output']
				if 'activation' in d:
					yield ['LEAKY']
		if len(d) == 1:
			if 'output' not in d:
				yield ['DROPOUT']
			else:
				if not flat:
					flat = True
					yield ['FLATTEN']
				yield ['conn', 0, 0,
				0, 0, 0, l, d['output']]
				l = d['output']