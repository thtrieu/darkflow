import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from darkflow.utils.box import BoundBox
from nms cimport NMS



@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def yolo_box_constructor(meta,np.ndarray[float] net_out, float threshold):

    cdef:
        float sqrt
        int C,B,S
        int SS,prob_size,conf_size
        int grid, b
        int class_loop

    
    sqrt =  meta['sqrt'] + 1
    C, B, S = meta['classes'], meta['num'], meta['side']
    boxes = []
    SS        =  S * S # number of grid cells
    prob_size = SS * C # class probabilities
    conf_size = SS * B # confidences for each grid cell

    cdef:
        float [:,::1] probs =  np.ascontiguousarray(net_out[0 : prob_size]).reshape([SS,C])
        float [:,::1] confs =  np.ascontiguousarray(net_out[prob_size : (prob_size + conf_size)]).reshape([SS,B])
        float [: , : ,::1] coords =  np.ascontiguousarray(net_out[(prob_size + conf_size) : ]).reshape([SS, B, 4])
        float [:,:,::1] final_probs = np.zeros([SS,B,C],dtype=np.float32)
        
    
    for grid in range(SS):
        for b in range(B):
            coords[grid, b, 0] = (coords[grid, b, 0] + grid %  S) / S
            coords[grid, b, 1] = (coords[grid, b, 1] + grid // S) / S
            coords[grid, b, 2] =  coords[grid, b, 2] ** sqrt
            coords[grid, b, 3] =  coords[grid, b, 3] ** sqrt
            for class_loop in range(C):
                probs[grid, class_loop] = probs[grid, class_loop] * confs[grid, b]
                #print("PROBS",probs[grid,class_loop])
                if(probs[grid,class_loop] > threshold ):
                    final_probs[grid, b, class_loop] = probs[grid, class_loop]
    
    
    return NMS(np.ascontiguousarray(final_probs).reshape(SS*B, C) , np.ascontiguousarray(coords).reshape(SS*B, 4))
