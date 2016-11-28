"""
file: /yolo/train.py
includes: parse(), batch(), and loss()
together they support the pipeline: 
    annotation -> minibatch -> loss evaluation -> training
namely,
parse() takes the path to annotation directory, returns the loaded cPickple dump
             that contains a list of parsed objects, each for an input image in trainset
batch() receive one such parsed objects, return feed value for net's input & output
             feed value for net's input will go to the input layer of net
             feed value for net's output will go to the loss layer of net
loss() basically build the loss layer of the net, namely,
            returns the corresponding placeholders for feed values of this loss layer
            as well as loss & train_op built from these placeholders and net.out
"""
import tensorflow.contrib.slim as slim
import cPickle as pickle
import tensorflow as tf
import numpy as np
import os

from utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from copy import deepcopy
from test import preprocess

def parse(FLAGS, meta):
    """
    Decide whether to parse the annotation or not, 
    If the parsed file is not already there, parse.
    """
    ext = '.parsed'
    history = os.path.join('tfnet', 'yolo', 'parse-history.txt');
    if not os.path.isfile(history):
        file = open(history, 'w')
        file.close()
    with open(history, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        labels = line[1:]
        if labels == meta['labels']:
            with open(line[0], 'rb') as f:
                return pickle.load(f)[0]

    # actual parsing
    ann = FLAGS.annotation
    if not os.path.isdir(ann):
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    print '\n{} parsing {}'.format(meta['model'], ann)
    dumps = pascal_voc_clean_xml(ann, meta['labels'])

    model = meta['model'].split('/')[-1]
    model = '.'.join(model.split('.')[:-1])
    save_to = os.path.join("tfnet", "yolo", model)
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

def batch(FLAGS, meta, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    S, B = meta['side'], meta['num']
    C, labels = meta['classes'], meta['labels']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(FLAGS.dataset, jpg)
    img = preprocess(path, allobj)

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
    conid = np.ones([S*S,B])
    cooid = np.zeros([S*S,B,4])
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
        #conid[obj[5], :] = [1.] * B
        cooid[obj[5], :, :] = [[1.] * 4] * B

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
        'probs':probs, 'confs':confs, 'coord':coord, 
        'proid':proid, 'conid':conid, 'cooid':cooid,
        'areas':areas, 'upleft':upleft, 'botright':botright
    }

    return inp_feed_val, loss_feed_val

def loss(net):
    """
    Takes net.out and placeholders value
    returned in batch() func above, 
    to build train_op and loss
    """
    # meta
    m = net.meta
    sprob = m['class_scale']
    sconf = m['object_scale']
    snoob = m['noobject_scale'] 
    scoor = m['coord_scale']
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S # number of grid cells

    print '{} loss hyper-parameters:'.format(m['model'])
    print '\tside    = {}'.format(m['side'])
    print '\tbox     = {}'.format(m['num'])
    print '\tclasses = {}'.format(m['classes'])
    print '\tscales  = {}'.format([sprob, sconf, snoob, scoor])

    size1 = [None, SS, C]
    size2 = [None, SS, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    _conid = tf.placeholder(tf.float32, size2)
    _cooid = tf.placeholder(tf.float32, size2 + [4])
    # material for loss calculation
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord,
        'proid':_proid, 'conid':_conid, 'cooid':_cooid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }

    # Extract the coordinate prediction from net.out
    coords = net.out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2 
    centers = coords[:,:,:,0:2] # [batch, SS, B, 2]
    floor = centers - (wh * .5) # [batch, SS, B, 2]
    ceil  = centers + (wh * .5) # [batch, SS, B, 2]

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft) 
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.mul(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])
    
    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.div(intersect, _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.mul(best_box, _confs)

    # take care of the weight terms
    weight_con = snoob * (1. - confs) + sconf * confs
    conid = tf.mul(_conid, weight_con)
    weight_coo = tf.concat(3, 4 * [tf.expand_dims(confs, -1)])
    cooid = tf.mul(_cooid, scoor * weight_coo)
    proid = sprob * _proid

    # flatten 'em all
    probs = slim.flatten(_probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)
    cooid = slim.flatten(cooid)
    true = tf.concat(1, [probs, confs, coord])
    wght = tf.concat(1, [proid, conid, cooid])

    print 'Building {} loss'.format(m['model'])
    loss = tf.pow(net.out - true, 2)
    loss = tf.mul(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return placeholders, loss
