"""
file: /yolo/train.py
includes: yolo_batch(), yolo_feed_dict() and yolo_loss()
together they support the pipeline: 
    annotation -> minibatch -> loss evaluation -> training
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from copy import deepcopy
from drawer import *

# ignore this function
def show(im, allobj, S, w, h, cellx, celly):
    for obj in allobj:
        a = obj[5] % S
        b = obj[5] / S
    	cx = a + obj[1]
    	cy = b + obj[2]
    	centerx = cx * cellx
    	centery = cy * celly
    	ww = obj[3] * w
    	hh = obj[4] * h
    	cv2.rectangle(im,
    		(int(centerx - ww/2), int(centery - hh/2)),
    		(int(centerx + ww/2), int(centery + hh/2)),
    		(0,0,255), 2)
    cv2.imshow("result", im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def yolo_batch(train_path, chunk, meta):
    """
    Takes a chunk of parsed annotations
    return placeholders for net's input
    correspond to this chunk
    """
    # meta
    S, B = meta['side'], meta['num']
    C, labels = meta['classes'], meta['labels']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = '{}{}'.format(train_path, jpg)
    img, allobj = yolo_preprocess(path, allobj)
    # img = yolo_preprocess(path)

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

    # Calculate placeholders' values
    probs = np.zeros([S*S,C])
    confs = np.zeros([S*S,B])
    coord = np.zeros([S*S,B,4])
    proid = np.zeros([S*S,C])
    conid = np.zeros([S*S,B])
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
        conid[obj[5], :] = [1.] * B
        cooid[obj[5], :, :] = [[1.] * 4] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft; 
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # Assemble the placeholders' value 
    tensors = [[probs], [confs] , [coord],
               [proid], [conid] , [cooid],
               [areas], [upleft], [botright]]
    
    return img, tensors

def yolo_feed_dict(net, x_batch, datum):
    return {
        net.probs : datum[0], net.confs  : datum[1],
        net.coord : datum[2], net.proid  : datum[3],
        net.conid : datum[4], net.cooid  : datum[5],
        net.areas : datum[6], net.upleft : datum[7], 
        net.botright : datum[8]
    }

def yolo_loss(net):
    """
    Takes net.out and placeholders -
    listed in feed_dict() func above, 
    to build net.train_op and net.loss
    """
    # meta
    m = net.meta
    sprob = m['class_scale']
    sconf = m['object_scale']
    snoob = m['noobject_scale'] 
    scoor = m['coord_scale']
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S # number of grid cells

    print 'Loss hyper-parameters:'
    print '\tside    = {}'.format(m['side'])
    print '\tbox     = {}'.format(m['num'])
    print '\tclasses = {}'.format(m['classes'])
    print '\tscales  = {}'.format([sprob, sconf, snoob, scoor])

    size1 = [None, SS, C]
    size2 = [None, SS, B]
    # target of regression
    net.probs = tf.placeholder(tf.float32, size1)
    net.confs = tf.placeholder(tf.float32, size2)
    net.coord = tf.placeholder(tf.float32, size2 + [4])
    # weights term for L2 loss
    net.proid = tf.placeholder(tf.float32, size1)
    net.conid = tf.placeholder(tf.float32, size2)
    net.cooid = tf.placeholder(tf.float32, size2 + [4])
    # material for loss calculation
    net.upleft = tf.placeholder(tf.float32, size2 + [2])
    net.botright = tf.placeholder(tf.float32, size2 + [2])
    net.areas = tf.placeholder(tf.float32, size2)

    # Extract the coordinate prediction from net.out
    coords = net.out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2 
    centers = coords[:,:,:,0:2] # [batch, SS, B, 2]
    floor = centers - (wh * .5) # [batch, SS, B, 2]
    ceil  = centers + (wh * .5) # [batch, SS, B, 2]

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, net.upleft) 
    intersect_botright = tf.minimum(ceil , net.botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.mul(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])
    
    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.div(intersect, net.areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.mul(best_box, net.confs)

    # take care of the weight terms
    weight_con = snoob*(1.-best_box) + sconf*best_box
    conid = tf.mul(net.conid, weight_con)
    weight_coo = tf.concat(3, 4 * [tf.expand_dims(best_box, -1)])
    cooid = tf.mul(net.cooid, scoor * weight_coo)
    proid = sprob * net.proid

    # flatten 'em all
    probs = slim.flatten(net.probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(net.coord)
    cooid = slim.flatten(cooid)
    true = tf.concat(1, [probs, confs, coord])
    wght = tf.concat(1, [proid, conid, cooid])
    
    net.loss = tf.pow(net.out - true, 2)
    net.loss = tf.mul(net.loss, wght)
    net.loss = tf.reduce_sum(net.loss, 1)
    net.loss = .5 * tf.reduce_mean(net.loss)

    optimizer = tf.train.RMSPropOptimizer(net.FLAGS.lr)
    gradients = optimizer.compute_gradients(net.loss)
    net.train_op = optimizer.apply_gradients(gradients)