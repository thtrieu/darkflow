"""
helpers of train, test
"""

import numpy as np
import cv2

labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]
default_models = ['yolo-full', 'yolo-new', 'yolo-small', 
                  'yolo-tiny', 'yolo-baby', 'tiny-yolo']

def labels(meta):
    model = meta['model'].split('/')[-1]
    model = '.'.join(model.split('.')[:-1])
    meta['name'] = model
    
    if model in default_models: 
        meta['labels'] = labels20
    else: 
        with open('labels.txt','r') as f:
            meta['labels'] = [l.strip() for l in f.readlines()]
    if len(meta['labels']) == 0: meta['labels'] = labels20

def is_inp(name): 
    return name[-4:] in ['.jpg','.JPG']

def show(im, allobj, S, w, h, cellx, celly):
    for obj in allobj:
        a = obj[5] % S
        b = obj[5] / S
    	cx = a + obj[1]
    	cy = b + obj[2]
    	centerx = cx * cellx
    	centery = cy * celly
    	ww = obj[3]**2 * w
    	hh = obj[4]**2 * h
    	cv2.rectangle(im,
    		(int(centerx - ww/2), int(centery - hh/2)),
    		(int(centerx + ww/2), int(centery + hh/2)),
    		(0,0,255), 2)
    cv2.imshow("result", im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show2(im, allobj):
    for obj in allobj:
        cv2.rectangle(im,
            (obj[1], obj[2]), 
            (obj[3], obj[4]), 
            (0,0,255),2)
    cv2.imshow('result', im)
    cv2.waitKey()
    cv2.destroyAllWindows()