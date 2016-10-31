from drawer import *
import cPickle as pickle
from copy import deepcopy
import subprocess
mult = 1.

def shuffle(train_path, file, expectC, S, batch, epoch):
	with open(file,'rb') as f:
		pick, data = pickle.load(f)
	C = len(pick)
	if C != expectC:
		exit("There is a mismatch between the model and the parsed annotations")
	size = len(data)
	print 'Dataset of total {}'.format(size)
	batch_per_epoch = int(size / batch)

	for i in range(epoch):
		print 'EPOCH {}'.format(i+1)
		# Shuffle data
		shuffle_idx = np.random.permutation(np.arange(size))
		for b in range(batch_per_epoch):
			for r in range(1):
				start_idx = b * batch
				end_idx = (b+1) * batch

				datum = list()
				x_batch = list()
				jpgs = list()
				try:
				# if True:
					for j in range(start_idx,end_idx):
						real_idx = shuffle_idx[j]
						this = data[real_idx]
						jpg = this[0]
						w, h, allobj_ = this[1]
						allobj = deepcopy(allobj_)
						flip = (r / 2)  + (r % 2) * (j % 2)
						flip = flip % 2

						path = '{}{}'.format(train_path, jpg)
						img, allobj = crop(path, allobj)

						if flip == 1: 
							img = img[:,:,::-1,:]

						img = [img]
						jpgs += [path]

						cellx = 1. * w / S
						celly = 1. * h / S
						for x in allobj:
							# cv2.rectangle(img[0], (x[1], x[2]), (x[3], x[4]), (0,0,255), 2)
							centerx = .5*(x[1]+x[3]) #xmin, xmax
							centery = .5*(x[2]+x[4]) #ymin, ymax
							if flip == 1:
								centerx = w - centerx
							cx = centerx / cellx
							cy = centery / celly
							x[3] = float(x[3]-x[1]) / w
							x[4] = float(x[4]-x[2]) / h
							x[3] = np.sqrt(x[3])
							x[4] = np.sqrt(x[4])
							x[1] = cx - np.floor(cx)
							x[2] = cy - np.floor(cy)
							x += [np.floor(cx)] 
							x += [np.floor(cy)]

						# if False:
						# 	for x in allobj:
						# 		cx = x[5] + x[1]
						# 		cy = x[6] + x[2]
						# 		centerx = cx * cellx
						# 		centery = cy * celly
						# 		ww = x[3] * x[3] * w
						# 		hh = x[4] * x[4] * h
						# 		cv2.rectangle(im,
						# 			(int(centerx - ww/2), int(centery - hh/2)),
						# 			(int(centerx + ww/2), int(centery + hh/2)),
						# 			(0,0,255), 2)

						# 	cv2.imshow("result", im)
						# 	cv2.waitKey()
						# 	cv2.destroyAllWindows()

						"""
						YOLO formulates the problem as a regression problem. Normally from the
						annotation, we can directly produce a target tensor to calculate the L2
						loss as (network_output - target)^2. But YOLO's L2 loss formulation is not
						that straightforward, namely the complication comes from its loss is selective:
						not penalizes all entries in the network_output, depending on what network_output
						looks like during training, moreover the loss also weights each term in the loss
						differently, e.g. coordinate term is weighted more than confidence terms, etc.

						To resolve this complication, I came up with a procedure that can calculate YOLO's
						loss function in two parts, all the operation in each part are tensor operations. The
						first part is done here during minibatch yielding, tensor operations are done on numpy
						tensors, the second part is done in decode() method inside tfnet.py, as tensorflow tensors.
						Why the seperation? I believe there are three reasons: 1. tensorflow tensors
						does not support member assignment, so any operation involving member assignment must be
						done as numpy tensors. 2. Efficiency: some operation are best to be done here than there.
						3. Inherent constraints in YOLO's formulation of the loss, please read the comming text
						for details.  
						
						The following text explains the next 11 tensors that I'll define
						They will be passed as placeholders into the network and serve as
						materials for calculating YOLO's loss. I look forward to suggestions
						on improving this (my) current approach.
						-----------------------------------------------------------------
		
						probs is the target class probability tensor
						confs1 and confs2 are confidence score of boxes 1 and boxes 2
						upleft are upper left corner coordinates of bounding boxes
						botright are bottom right corner coordinates of bounding boxes
						So far, probs, confs1, confs2, upleft, botright constitutes the target 
						of regression, why do we need the ___id tensors?

						You know from the paper that only grid cells that are responsible for 
						correct prediction are penalized (by an L2 loss), so not all entries in
						the above tensors should take part in the loss calculation, furthermore 
						according to the paper, coordinates terms in the loss should be weighted more 
						than the other terms, and of two boxes that each grid cell predicts, one with better 
						IOU should be weighted differently than the other.

						These __id tensors are meant to solve the above complication. They act as weights
						and will be set to appropriate value either in data.py (as numpy tensors, during the 
						batch generating phase (this file)) or in tfnet.py (as tensorflow tensors, during the 
						loss calculation phase). For example, if an entry should not affect the loss, its 
						corresponding weight will be set to zero, if an entry correspond to coordinate loss, 
						the weight should be 5.0, so on.

						proid will weight probs, and its final value is set here in data.py
						conid1 weights confs1
						conid2 weights confs2
						cooid1 weights coordinate of box1
						cooid2 weights coordinate of box2

						conid1, conid2, cooid1, cooid2's values are initialised in data.py and set to correct value 
						in tfnet.py. Why? because we only know their correct value when IOU of each predicted box 
						with the target are calculated, i.e. the forward pass must be done before this. 
						"""
						probs = np.zeros([S*S,C])
						confs = np.zeros([S*S,2])
						coord = np.zeros([S*S,2,4])
						proid = np.zeros([S*S,C])
						conid = np.zeros([S*S,2])
						cooid1 = cooid2 = np.zeros([S*S,1,4])
						prear = np.zeros([S*S,4])
						for x in allobj:
							at = int(x[6] * S + x[5])
							probs[at, :] = [0.] * C
							probs[at, pick.index(x[0])] = 1.
							proid[at, :] = [1] * C
							coord[at, 0, :] = x[1:5]
							coord[at, 1, :] = x[1:5]
							scale = .5 * S
							prear[at,0] = x[1] - x[3]**2 * scale # xleft
							prear[at,1] = x[2] - x[4]**2 * scale # yup
							prear[at,2] = x[1] + x[3]**2 * scale # xright
							prear[at,3] = x[2] + x[4]**2 * scale # ybot
							confs[at, :] = [1.] * 2
							conid[at, :] = [1.] * 2
							cooid1[at, 0, :] = [1.] * 4
							cooid2[at, 0, :] = [1.] * 4
						upleft   = np.expand_dims(prear[:,0:2], 1) # 49 x 1 
						botright = np.expand_dims(prear[:,2:4], 1)

					# Finalise the placeholders' values
						probs = probs.reshape([-1]) # true_class
						confs1 = confs[:,0]
						confs2 = confs[:,1]
						coord = coord.reshape([-1]) # true_coo
						upleft   = np.concatenate([upleft]*2,1)
						botright = np.concatenate([botright]*2,1)
						proid = proid.reshape([-1]) # class_idtf
						conid1 = conid[:,0]
						conid2 = conid[:,1]
						cooid1 = cooid1
						cooid2 = cooid2

					# Assemble the placeholders' value 
						new = [
							[probs], [confs1], [confs2], [coord],
							[upleft], [botright],
							[proid], [conid1], [conid2], [cooid1], [cooid2]
						]
						if datum == list():
							datum = new
							x_batch = img
						else:
							x_batch += img
							for i in range(len(datum)):
								datum[i] = np.concatenate([datum[i], new[i]])

						if False:
							here = 0
							names = list()
							while here + C < S*S*C:
								consider = probs[here:here+C]
								if (np.sum(consider) > 0.5):
									names += [pick[np.argmax(consider)]]
								here += C
							print '{} : {}'.format(jpg, names)


					x_batch = np.concatenate(x_batch, 0)
					yield (x_batch, datum)
				except:
					print 'Random scale/translate sends object(s) out of bound'
					continue
