from yolo.train import *

def convl(l, x, name):
    if l.pad < 0: # figure the pad out
        size = np.int(x.get_shape()[1])
        expect = -(l.pad + 1) * l.stride 
        expect += l.size - size
        padding = [expect / 2, expect - expect / 2]
        if padding[0] < 0: padding[0] = 0
        if padding[1] < 0: padding[1] = 0
    else:
        padding = [l.pad, l.pad]
    l.pad = 'VALID'
    x = tf.pad(x, [[0, 0], padding, padding, [0, 0]])
    x = tf.nn.conv2d(x, l.weights, 
        padding = l.pad, name = name,
        strides=[1, l.stride, l.stride, 1])
    # if l.batch_norm == 1: x = slim.batch_norm(x)
    # else: x = tf.nn.bias_add(x, l.b)
    return tf.nn.bias_add(x, l.biases)

def bnorm(l, x, name):
    return x

def dense(l, x, name):
    return tf.nn.xw_plus_b(x, l.weights, l.biases, name = name)
    
def maxpool(l, x, name):
    l.pad = 'VALID'
    return tf.nn.max_pool(x, padding = l.pad,
        ksize = [1,l.size,l.size,1], name = name, 
        strides = [1,l.stride,l.stride,1])

def flatten(x, name):
    x = tf.transpose(x, [0,3,1,2])
    return slim.flatten(x, scope = name)

def leaky(x, name):
    return tf.maximum(.1*x, x, name = name)

def dropout(x, drop, name):
    return tf.nn.dropout(x, drop, name = name)
