import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from darkflow.utils.box import BoundBox
from nms cimport NMS
from nms cimport NMS9000

#expit
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float expit_c(float x):
    cdef float y= 1/(1+exp(-x))
    return y

#MAX
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float max_c(float a, float b):
    if(a>b):
        return a
    return b

#SOFTMAX!
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def softmax_classes(Classes, subC, write_threshold, row,col,box_loop):
    cdef:
      float arr_max=0,sum=0

    for class_loop in subC:
        arr_max=max_c(arr_max,Classes[row,col,box_loop,class_loop])
    
    for class_loop in subC:
        Classes[row,col,box_loop,class_loop]=exp(Classes[row,col,box_loop,class_loop]-arr_max)
        sum+=Classes[row,col,box_loop,class_loop]
    
    for class_loop in subC:
        tempc = Classes[row,col,box_loop,class_loop]/sum # * Bbox_pred[row,col,box_loop,4] 
        # This threshold determines what hyponyms splits could warrant further exploration
        if(tempc > write_threshold): # If below hierarchy threshold, will not hit this node ever 
          Classes[row,col,box_loop,class_loop] = tempc 
"""
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
"""


#CREATE SOFTMAX AND CONDITIONAL TREE FROM NET OUTPUT 
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def conditional_softmax_tree(Classes, parent_index, hierarchy_threshold, tree_dict, row,col,box_loop):
    cdef:
      float arr_max=0,sum=0
      np.intp_t group_size, offset
    hyponyms = tree_dict[parent_index] # Find the children of this parent
    group_size = len(hyponyms) 
    offset = hyponyms[0] 

    softmax_classes(Classes, range(offset, offset+group_size), hierarchy_threshold, row,col,box_loop)

    for class_loop in range(offset, offset+group_size): 
      if (Classes[row,col,box_loop,class_loop] > hierarchy_threshold): 
        if class_loop in tree_dict:
          conditional_softmax_tree(Classes, class_loop, hierarchy_threshold, tree_dict, row,col,box_loop)


#FOLLOW TREE TO FIND MOST SPECIFIC PREDICTION BEFORE THRESHOLD
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def find_top_prediction(Classes, parent_index, hierarchy_threshold, tree_dict, row,col,box_loop):
    cdef:
      float arr_max=0,sum=0
      int index_max=0
      np.intp_t group_size, offset
    hyponyms = tree_dict[parent_index] # Find the children of this parent
    group_size = len(hyponyms) 
    offset = hyponyms[0] 

    for class_loop in range(offset, offset+group_size):
        if Classes[row, col, box_loop, class_loop] > arr_max:
           arr_max = Classes[row, col, box_loop, class_loop]
           index_max = class_loop

    if (arr_max > hierarchy_threshold): # Progress down the tree until value not above threshold 
      if index_max in tree_dict and len(tree_dict[index_max]) > 0: # Hit leaf node of tree
        return find_top_prediction(Classes, index_max, hierarchy_threshold, tree_dict, row,col,box_loop) 
      else:
        return index_max
    else:
      return parent_index
        

#BOX CONSTRUCTOR
@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def box_constructor(meta,np.ndarray[float,ndim=3] net_out_in):
    cdef:
        np.intp_t H, W, _, C, B, row, col, box_loop, class_loop
        np.intp_t row1, col1, box_loop1,index,index2, top_pred_index
        float threshold = meta['thresh']
        float hierarchy_threshold = meta['hier_thresh'] 
        float tempc,arr_max=0,sum=0
        double[:] anchors = np.asarray(meta['anchors'])
        list boxes = list()

    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']
    if 'tree_dict' in meta:
      tree_dict = meta['tree_dict']
    if 'map_array' in meta:
      map_array = meta['map_array']
    
    cdef:
        float[:, :, :, ::1] net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B])
        float[:, :, :, ::1] Classes = net_out[:, :, :, 5:]
        float[:, :, :, ::1] Bbox_pred =  net_out[:, :, :, :5]
        float[:, :, :, ::1] probs = np.zeros((H, W, B, C), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 4] = expit_c(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H

                # To predict an object, YOLO needs its object score to be above threshold
                if(Bbox_pred[row,col,box_loop,4] > threshold):
                    if 'tree_dict' in meta:
                        # SEE get_region_boxes() in darknet as a reference
                        # Softmax and make all probabilities conditional

                        # This section lets us run detection on just the 200 object classes
                        if 'map_array' in meta:
                            softmax_classes(Classes, list(meta['map_array']), 0.0, row,col,box_loop)
                            for index in meta['map_array']:
                                prob = Classes[row,col,box_loop,index]*Bbox_pred[row,col,box_loop,4] 
                                if prob > threshold:
                                    probs[row,col,box_loop,index] = prob
                        else:
                            # Else we produce YOLO9000 results 
                            conditional_softmax_tree(Classes, -1, hierarchy_threshold, tree_dict, row,col,box_loop)
                            # We set the last detection that passes threshold as the object score for this box
                            top_pred_index = find_top_prediction(Classes, -1, hierarchy_threshold, tree_dict, row,col,box_loop) 
                            probs[row,col,box_loop,top_pred_index] = Bbox_pred[row,col,box_loop,4] 

                            # This sets the last element in the class list for easy NMS box to box checking
                        probs[row,col,box_loop,C-1] = Bbox_pred[row,col,box_loop,4] 
                    else:
                        softmax_classes(Classes, range(C), 0.0, row,col,box_loop)

                        for class_loop in range(C):
                            tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]                    
                            if(tempc > threshold):
                                probs[row, col, box_loop, class_loop] = tempc
    
    
    #NMS                    
    if 'tree_dict' not in meta:
        return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))
    else:
        return NMS9000(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))
