from utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from test import preprocess
from copy import deepcopy
import cPickle as pickle
import numpy as np
import os 

def expit(x):
	return 1. / (1. + np.exp(-x))

def parse(self, exclusive = False):
    """
    Decide whether to parse the annotation or not, 
    If the parsed file is not already there, parse.
    """
    meta = self.meta
    ext = '.parsed'
    history = os.path.join('net', 'yolo', 'parse-history.txt');
    if not os.path.isfile(history):
        file = open(history, 'w')
        file.close()
    with open(history, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        labels = line[1:]
        if labels == meta['labels']:
            if os.path.isfile(line[0]):
                with open(line[0], 'rb') as f:
                    return pickle.load(f)[0]

    # actual parsing
    ann = self.FLAGS.annotation
    if not os.path.isdir(ann):
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    print '\n{} parsing {}'.format(meta['model'], ann)
    dumps = pascal_voc_clean_xml(ann, meta['labels'], exclusive)

    save_to = os.path.join('net', 'yolo', meta['name'])
    while True:
        if not os.path.isfile(save_to + ext): break
        save_to = save_to + '_'
    save_to += ext

    with open(save_to, 'wb') as f:
        pickle.dump([dumps], f, protocol = -1)
    with open(history, 'a') as f:
        f.write('{} '.format(save_to))
        f.write(' '.join(meta['labels']))
        f.write('\n')
    print 'Result saved to {}'.format(save_to)
    return dumps


def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    labels = meta['labels']
    
    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H: return None, None
        obj[3] = float(obj[3]-obj[1]) / w
        obj[4] = float(obj[4]-obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]

    #show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])
    for obj in allobj:
        probs[obj[5], :, :] = [[0.]*C] * B
        probs[obj[5], :, labels.index(obj[0])] = 1.
        proid[obj[5], :, :] = [[1.]*C] * B
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * W # xleft
        prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * H # yup
        prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * W # xright
        prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * H # ybot
        confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft; 
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright
    }

    return inp_feed_val, loss_feed_val

def shuffle(self):
    batch = self.FLAGS.batch
    data = self.parse()
    size = len(data)

    print 'Dataset of {} instance(s)'.format(size)
    if batch > size: self.FLAGS.batch = batch = size
    batch_per_epoch = int(size / batch)

    for i in range(self.FLAGS.epoch):
        shuffle_idx = perm(np.arange(size))
        for b in range(batch_per_epoch):
            # yield these
            x_batch = list()
            feed_batch = dict()

            for j in range(b*batch, b*batch+batch):
                train_instance = data[shuffle_idx[j]]
                inp, new_feed = _batch(self, train_instance)

                if inp is None: continue
                x_batch += [np.expand_dims(inp, 0)]

                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed_batch.get(key, 
                        np.zeros((0,) + new.shape))
                    feed_batch[key] = np.concatenate([ 
                        old_feed, [new] 
                    ])      
            
            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed_batch
        
        print('Finish {} epoch(es)'.format(i + 1))

