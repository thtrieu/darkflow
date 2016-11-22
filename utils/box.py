import numpy as np

class BoundBox:
    def __init__(self, classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.c = 0
        self.class_num = 0
        self.probs = np.zeros((classes,))

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2.;
    l2 = x2 - w2/2.;
    left = max(l1, l2)
    r1 = x1 + w1/2.;
    r2 = x2 + w2/2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 or h < 0):
         return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

def prob_compare(boxa,boxb):
    if (boxa.probs[boxa.class_num] < boxb.probs[boxb.class_num]):
        return 1
    elif(boxa.probs[boxa.class_num] == boxb.probs[boxb.class_num]):
        return 0
    else:
        return -1