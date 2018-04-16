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
        # print("CURRENT OBJECT: ", obj)

        centerx = .5 * (obj[1] + obj[3])  # xmin, xmax
        centery = .5 * (obj[2] + obj[4])  # ymin, ymax

        # print("centrex: ", centerx, " centrey: ", centery)
        # print("width: ", (obj[3] - obj[1]), " height: ", (obj[4] - obj[2]))

        # Cell x and Cell y are the width and height of the cells in the image.
        cx = centerx / cellx
        cy = centery / celly
        # print("Centre point x:{}, y:{}".format(centerx, centery))
        # print("cx:{}, cy:{}".format(cx,cy))
        # print("WTF: ", obj[4])

        if cx >= S or cy >= S: return None, None
        # print("w: {}, h: {}".format(w, h))
        # Normalise the value and turn them into darknet format. (YOPO -21)
        obj[3] = float(obj[3] - obj[1]) / w # w of image (normalised)
        obj[4] = float(obj[4] - obj[2]) / h # h of image (normalised)
        # print("Check Sqaure root func, {} {}".format(obj[3], obj[4]))
        #
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        # print("Check Sqaure root func, AFTER: {} {}".format(obj[3], obj[4]))
        # print("XXX", obj[1])
        # print("YYY", obj[2])
        # Offset inside cell!
        obj[1] = cx - np.floor(cx)  # off set x for a given cell
        obj[2] = cy - np.floor(cy)  # off set y for a given cell

        # Normalise Angle

        obj[5] = obj[5] / 360

        # print("AFTER: XXX", obj[1])
        # print("AFTER: YYY", obj[2])
        # print("BEN: ", int(np.floor(cy) * S + np.floor(cx)))
        # print("Benfinal: ", obj[1:5])
        obj += [int(np.floor(cy) * S + np.floor(cx))]

        # print(obj)

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    # print(C)
    # print(B)
    # print(labels)

    probs = np.zeros([S * S, C])
    confs = np.zeros([S * S, B])
    coord = np.zeros([S * S, B, 5])
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
        probs[obj[6], :] = [0.] * C
        # Set the confidence score of that class to 1 because it's label so you it's 100% that class.
        probs[obj[6], labels.index(obj[0])] = 1.
        #
        proid[obj[6], :] = [1] * C #
        # print("BCUNT", obj[1:6])
        # Copies Box Coordinates to it's cell and creates three copies of the same box.
        coord[obj[6], :, :] = [obj[1:6]] * B

        # Normalise Angle by 360
        # print(coord[obj[6], :, :])
        # coord = coord[obj[6], :, 4]
        # print(coord[obj[6], :, :])

        prear[obj[6], 0] = obj[1] - obj[3] ** 2 * .5 * S  # xleft
        prear[obj[6], 1] = obj[2] - obj[4] ** 2 * .5 * S  # yup
        prear[obj[6], 2] = obj[1] + obj[3] ** 2 * .5 * S  # xright
        prear[obj[6], 3] = obj[2] + obj[4] ** 2 * .5 * S  # ybot
        confs[obj[6], :] = [1.] * B

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
