import numpy as np
import cv2
import os

labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]
    
voc_models = ['yolo-full', 'yolo-tiny', 'yolo-small',  # <- v1
              'yolov1', # <- v1.1 
              'tiny-yolo-voc'] # <- v2

coco_models = ['tiny-coco', 'yolo-coco',  # <- v1.1
               'yolo', 'tiny-yolo'] # <- v2

def labels(meta):
    model = meta['model'].split('/')[-1]
    model = '.'.join(model.split('.')[:-1])
    meta['name'] = model
    
    if model in voc_models: 
        meta['labels'] = labels20
    else:
        file = 'labels.txt'
        if model in coco_models: 
            file = os.path.join('cfg','coco.names')
        with open(file, 'r') as f:
            meta['labels'] = [l.strip() for l in f.readlines()]

    if len(meta['labels']) == 0: meta['labels'] = labels20

def is_inp(self, name): 
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