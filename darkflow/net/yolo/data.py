from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from .predict import preprocess
# from .misc import show
from copy import deepcopy
import pickle
import numpy as np
import os


def parse(self, exclusive=False):
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
    jpg = chunk[0];
    w, h, allobj_ = chunk[1]

    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    #here be dragons
    #todo: turn this bollocks off?
    img = self.preprocess(path, allobj)
    # print("ALL object: ", allobj)
    # Calculate regression target
    # Which cell is the object centre in.

    # Find width a of single cell.
    # print("Cell size", S)
    cellx = 1. * w / S
    celly = 1. * h / S
    print("BATCH INFORMATION HERE: ", cellx, celly, jpg)
    # YOPO Converting all images and labels to darknet format.
    for obj in allobj:

        # min and max values for the network i.e BBox vertices

        obj[1] = obj[1] / cellx # xmin
        obj[3] = obj[3] / cellx # xmax
        obj[2] = obj[2] / celly # ymin
        obj[4] = obj[4] / celly # ymax

        print("Object Vertices, xmin: {}, xmax: {}, ymin: {}, ymax: {}".format(obj[1], obj[2], obj[3], obj[4]))

        #  if this is true, everything has gone wrong and it's time eject out of here.
        if obj[1] >= S or obj[2] >= S or obj[3] >= S or obj[4] >= S: return None, None


    probs = np.zeros([S * S, C])
    confs = np.zeros([S * S, B])
    coord = np.zeros([S * S, B, 4])
    proid = np.zeros([S * S, C])
    prear = np.zeros([S * S, 4])
    iou = np.zeros([S * S, B])
    image_tens = np.zeros([2])

    image_tens[0] = w
    image_tens[1] = h
    # 6 is the prob score?
    # print("************************")
    for obj in allobj:
        # print("obj", obj)
        # All to do with confidence score of the boxes.
        probs[obj[5], :] = [0.] * C
        # Set the confidence score of that class to 1 because it's label so you it's 100% that class.
        probs[obj[5], labels.index(obj[0])] = 1.
        #
        proid[obj[5], :] = [1] * C #

        # Copies Box Coordinates to it's cell and creates three copies of the same box.
        coord[obj[5], :, :] = [obj[1:5]] * B

        # todo??? NOT SURE ABOUT THIS ONE HERE!!!
        # Change these to just xmin, xmax, ymin, ymax
        prear[obj[5], 0] = obj[1] - obj[3] ** 2 * .5 * S  # xleft
        prear[obj[5], 1] = obj[2] - obj[4] ** 2 * .5 * S  # yup
        prear[obj[5], 2] = obj[1] + obj[3] ** 2 * .5 * S  # xright
        prear[obj[5], 3] = obj[2] + obj[4] ** 2 * .5 * S  # ybot
        confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)
    wh = botright - upleft;
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)
    # angle = np.concatenate()

    # value for placeholder at input layer
    inp_feed_val = img

    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'image': image_tens,
        'iou': iou
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

            for j in range(b * batch, b * batch + batch):
                train_instance = data[shuffle_idx[j]]
                try:
                    inp, new_feed = self._batch(train_instance)
                except ZeroDivisionError:
                    print("This image's width or height are zeros: ", train_instance[0])
                    print('train_instance:', train_instance)
                    print('Please remove or fix it then try again.')
                    raise

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
