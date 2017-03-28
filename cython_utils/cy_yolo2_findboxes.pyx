import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from utils.box import BoundBox


#OVERLAP
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float overlap_c(float x1, float w1 , float x2 , float w2):
    cdef:
        float l1,l2,left,right
    l1 = x1 - w1 /2.
    l2 = x2 - w2 /2.
    left = max(l1,l2)
    r1 = x1 + w1 /2.
    r2 = x2 + w2 /2.
    right = min(r1, r2)
    return right - left;

#BOX INTERSECTION
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_intersection_c(float* a, float* b):
    cdef:
        float w,h,area
    w = overlap_c(a[0], a[2], b[0], b[2])
    h = overlap_c(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0: return 0
    area = w * h
    return area

#BOX UNION
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_union_c(float* a, float* b):
    cdef:
        float i,u
    i = box_intersection_c(a, b)
    u = a[2] * a[3] + b[2] * b[3] -i
    return u


#BOX IOU
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_iou_c(float* a, float* b):
    return box_intersection_c(a, b) / box_union_c(a, b);


#expit
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float expit_c(float x):
    cdef float y= 1/(1+exp(-x))
    return y


#SOFTMAX!
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void _softmax_c(float* x, int classes):
    cdef:
        float sum = 0
        np.intp_t k
        float arr_max = 0
    for k in range(classes):
        arr_max = max(arr_max,x[k])
    
    for k in range(classes):
        x[k] = exp(x[k]-arr_max)
        sum += x[k]

    for k in range(classes):
        x[k] = x[k]/sum
        
        

#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def box_constructor(meta,np.ndarray[float,ndim=3] net_out_in):
    cdef:
        np.intp_t H, W, _, C, B, row, col, box_loop, class_loop
        np.intp_t row1, col1, box_loop1
        np.intp_t Box_param_dim
        float  threshold = meta['thresh']
        float tempc
        double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()

    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']
    
    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, 5:]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, :5]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)
        float* Class_Ptr = &Classes[0, 0, 0, 0]
        float* Bbox_Ptr = &Bbox_pred[0, 0, 0, 0]
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                Bbox_pred[row, col, box_loop, 4] = expit_c(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H
                _softmax_c(Class_Ptr + row*W*B*C + col*B*C + box_loop*C, C)
                for class_loop in range(C):
                    tempc = Class_Ptr[row*W*B*C + col*B*C + box_loop*C + class_loop] * Bbox_pred[row, col, box_loop, 4]
                    if(tempc > threshold):
                        probs[row, col, box_loop, class_loop] = tempc



               
    
    

    #NMS
    #print "NMS"
    Box_param_dim = Bbox_pred.shape[3]
    for class_loop in range(C): #inital class loop
        for row in range(H):
            for col in range(W):
                for box_loop in range(B):
                    if probs[row, col, box_loop, class_loop] == 0: continue
                    for row1 in range(H):
                        for col1 in range(W):
                            for box_loop1 in range(B):
                                if probs[row1, col1, box_loop1, class_loop] == 0: continue
                                if box_iou_c(Bbox_Ptr + row*W*B*Box_param_dim + col*B*Box_param_dim + box_loop*Box_param_dim, Bbox_Ptr + row1*W*B*Box_param_dim + col1*B*Box_param_dim + box_loop1*Box_param_dim) >= 0.4:
                                    probs[row1, col1, box_loop1, class_loop] = 0
                    #append the survivor 
                    #print "Survivor"
                    bb = BoundBox(C)
                    bb.class_num = class_loop;
                    bb.x = Bbox_pred[row, col, box_loop, 0]
                    bb.y = Bbox_pred[row, col, box_loop, 1]
                    bb.w = Bbox_pred[row, col, box_loop, 2]
                    bb.h = Bbox_pred[row, col, box_loop, 3]
                    bb.c = Bbox_pred[row, col, box_loop, 4]
                    bb.probs = np.asarray(probs[row, col, box_loop, :])
                    boxes.append(bb)

    return boxes
