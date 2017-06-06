from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from .predict import preprocess
# from .misc import show
from copy import deepcopy
import pickle
import numpy as np
import os 

def parse(self, exclusive = False):
    meta = self.meta
    ext = '.parsed'
    ann = self.FLAGS.annotation
    if not os.path.isdir(ann):
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    print('\n{} parsing {}'.format(meta['model'], ann))
    dumps = pascal_voc_clean_xml(ann, meta['labels'], exclusive)
    return dumps


def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    S, B = meta['side'], meta['num']
    C, labels = meta['classes'], meta['labels']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / S
    celly = 1. * h / S
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= S or cy >= S: return None, None
        obj[3] = float(obj[3]-obj[1]) / w
        obj[4] = float(obj[4]-obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery
        obj += [int(np.floor(cy) * S + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([S*S,C])
    confs = np.zeros([S*S,B])
    coord = np.zeros([S*S,B,4])
    proid = np.zeros([S*S,C])
    prear = np.zeros([S*S,4])
    for obj in allobj:
        probs[obj[5], :] = [0.] * C
        probs[obj[5], labels.index(obj[0])] = 1.
        proid[obj[5], :] = [1] * C
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * S # xleft
        prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * S # yup
        prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * S # xright
        prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * S # ybot
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

    print('Dataset of {} instance(s)'.format(size))
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
                inp, new_feed = self._batch(train_instance)

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

