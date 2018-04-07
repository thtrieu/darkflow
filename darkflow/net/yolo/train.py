import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf

from darkflow.net.yopo.calulating_IOU import intersection_over_union, Rectangle
from .misc import show
import numpy as np
import os
import pprint as pp
from copy import deepcopy
import math


def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta Ground Truth
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    S, B, C = m['side'], m['num'], m['classes']

    print("Number of Grid Cells", S)

    SS = S * S  # number of grid cells

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tside    = {}'.format(m['side']))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    # Any 49, 8 So for each cell that class is it?
    size1 = [None, SS, C]
    # BBox confidence Scoring Tensor
    size2 = [None, SS, B]

    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [5])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    # _areas = tf.placeholder(tf.float32, size2)
    # _upleft = tf.placeholder(tf.float32, size2 + [2])
    # _botright = tf.placeholder(tf.float32, size2 + [2])
    _image = tf.placeholder(tf.float32, [None, 2])

    # iou = tf.placeholder(tf.float32, size2)
    iou = tf.placeholder(tf.float32, size2)

    # return the below placeholders
    # print("Shape of Top left: {}, Bot Right {}".format(_upleft, _botright))

    self.placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'image': _image, 'iou': iou
    }

    # print(self.placeholders[0][0][0])
    # tf.Tensor.eval(self,)
    # Extract the coordinate prediction from net.out
    coords = net_out[:, SS * (C + B):]
    # Make coords array back into a tensor.
    coords = tf.reshape(coords, [-1, SS, B, 5])

    iou = tf.py_func(calculate_iou, [_image, _coord, coords, iou], tf.float32)
    iou = tf.reshape(iou, [-1, SS, B])

    # coords_print = tf.py_func(yopo_print, [coords], tf.float32)
    # coords_print.set_shape(coords.get_shape())

    #
    wh = tf.pow(coords[:, :, :, 2:4], 2) * S  # unit: grid cell
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]  # unit: grid cell^2
    centers = coords[:, :, :, 0:2]  # [batch, SS, B, 2]

    # Print a box - Might need un-normalised it.
    # new_wh = tf.py_func(print_box, [centers, wh], tf.float32)
    # new_wh.set_shape(wh.get_shape())

    floor = centers - (wh * .5)  # [batch, SS, B, 2]
    ceil = centers + (wh * .5)  # [batch, SS, B, 2]

    # output = tf.Print(_areas, [_areas], "_area tensor")
    # _area_output = tf.py_func(yopo_print, [area_pred], tf.float32)
    # _area_output.set_shape(area_pred.get_shape())
    # print("OUTPUT:", output)

    # calculate the intersection areas
    # intersect_upleft = tf.maximum(floor, _upleft)
    # intersect_botright = tf.minimum(ceil, _botright)
    # intersect_wh = intersect_botright - intersect_upleft
    # intersect_wh = tf.maximum(intersect_wh, 0.0)
    # intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])
    # intersect_new = tf.py_func(printTensor, [intersect], tf.float32)
    # intersect_new.set_shape(intersect.get_shape())

    # calculate the best IOU, set 0.0 confidence for worse boxes
    # iou = tf.truediv(intersect_new, _areas + area_pred - intersect_new, "IOU")

    # print('IOU shape: ', iou)

    # new_iou = tf.py_func(testFunc, [true, net_out], tf.float32)
    # new_iou.set_shape(iou.get_shape())

    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    # Class Probs * box Confidence
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(5 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    proid = sprob * _proid

    # flatten 'em all
    probs = slim.flatten(_probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)
    cooid = slim.flatten(cooid)
    #
    # coord_list = tf.py_func(printTensor, [coord, coords], tf.float32)
    # coord_list.set_shape(coord.get_shape())
    # coord_list[1].set_shape(coords.get_shape())

    self.fetch += [probs, confs, conid, cooid, proid]
    true = tf.concat([probs, confs, coord], 1)

    # new_iou = tf.py_func(testFunc, [true, net_out], tf.float32)
    # new_iou.set_shape(iou.get_shape())

    wght = tf.concat([proid, conid, cooid], 1)
    print('Building {} loss'.format(m['model']))

    loss = tf.pow(net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)


def printTensor(tensor):
    for x in tensor:
        print("Best Box ", x)
        print("length: ", len(x))
    return tensor


# This function is a custom graph operation writen in python.
def calculate_iou(image_tens, gt_tensor, net_out_tensor, iou):
    print("Start calculate_iou")
    image_index = 0
    for ground_truth, net_out_tensor in zip(gt_tensor, net_out_tensor):
        # todo
        ground_truth = deepcopy(ground_truth)
        net_out_tensor = deepcopy(net_out_tensor)
        cell_index = 0
        S = math.sqrt(len(ground_truth))
        image_width = image_tens[image_index][0]
        image_height = image_tens[image_index][1]
        print("\nImage", image_index, " W: ", image_width, " H: ", image_height, " S: ", S)
        for ground_truth_cell, net_out_cell in zip(ground_truth, net_out_tensor):
            # print("Ground Truth Tensor:", ground_truth, "Network out Tensor:", net_out_tensor)
            cell_box_index = 0

            empty = 5
            for c in ground_truth_cell[0]:
                if c == 0:
                    empty = empty - 1
                else:
                    break

            if empty == 0:
                cell_index = cell_index + 1
                continue

            for ground_truth_box, net_out_box in zip(ground_truth_cell, net_out_cell):

                # # print("*********************** ")
                # print("Ground Truth")
                # print(ground_truth_box)
                # print("net out box")
                # print(net_out_box)

                # Ground Truth Tensor
                # TODO FIX GRID CELLS!!!
                cell_x = cell_index % S
                cell_y = math.floor(cell_index / S)

                cell_width = (image_width / S)
                cell_height = (image_height / S)

                # print("Cell width: ", cell_width, " Cell height: ", cell_height)
                # print("Cell offset x: ", ground_truth_cell[0][0], " Cell offset y: ", ground_truth_cell[0][1])

                centre_x = (cell_width * cell_x) + (ground_truth_box[0] * cell_width)
                centre_y = (cell_height * cell_y) + (ground_truth_box[1] * cell_height)

                gt_width = (ground_truth_box[2] ** 2) * image_width
                gt_height = (ground_truth_box[3] ** 2) * image_height

                print("Angled Ground Truth", ground_truth_box[4])
                gt_angle = ground_truth_box[4] * 360


                # print("NEW At ", cell_index, " CX: ", cell_x, " CY: ", cell_y, " X: ", centre_x, " Y: ", centre_y,
                #       " W: ", gt_width, " H: ", gt_height)
                # print("Cell Number", cell_index, " CX: ", cell_x, " CY: ", cell_y)

                # Create ground truth Tensor
                ground_truth_rec = Rectangle(centre_x, centre_y, gt_width, gt_height, gt_angle)
                # print("Ground Truth Rectangle", ground_truth_rec, "\n")

                # Network out tensor

                # print("Output Network Tensor")

                # print("Cell offset x: ", net_out_box[0], " Cell offset y: ", net_out_box[1])

                out_net_centre_x = (cell_width * cell_x) + (net_out_box[0] * cell_width)
                out_net_centre_y = (cell_height * cell_y) + (net_out_box[1] * cell_height)

                out_net_width = (net_out_box[2] ** 2) * image_width
                out_net_height = (net_out_box[3] ** 2) * image_height


                net_out_angle = net_out_box[4]
                net_out_angle = net_out_angle * 360
                print("RAW: Angle Network output", net_out_box[4])

                print("GROUND TRUTH ANGLE", gt_angle, "IOU Calculation Network", net_out_angle)
                #
                # print("Output At ", cell_index, " CX: ", cell_x, " CY: ", cell_y, " X: ", out_net_centre_x, " Y: ",
                #       out_net_centre_y, " W: ", out_net_width, " H: ", out_net_height, 'angle')

                # Create ground truth Tensor
                out_net_rec = Rectangle(out_net_centre_x, out_net_centre_y, out_net_width, out_net_height,
                                        net_out_angle)

                print("Ground Truth Rectangle", ground_truth_rec)
                print("Output Network Rectangle", out_net_rec)

                iou_val = intersection_over_union(ground_truth_rec, out_net_rec)

                # if cell_index == 51:
                print("IOU for box {}: {} \n\n".format(cell_box_index, iou_val))


                # print("IOU for box ", cell_box_index, ": ", iou_val)
                iou[image_index][cell_index][cell_box_index] = iou_val

                cell_box_index = cell_box_index + 1

            cell_index = cell_index + 1
        image_index = image_index + 1
    print("End calculate_iou")
    return iou
