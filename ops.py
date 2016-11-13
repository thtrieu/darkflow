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
    x = tf.nn.conv2d(x, l.p['kernel'], padding = l.pad, 
        name = name,strides=[1, l.stride, l.stride, 1])
    if l.batch_norm:
        x = batchnorm(l, x, '{}-bnorm'.format(name))
    return tf.nn.bias_add(x, l.p['biases'])

def batchnorm(l, x, name):
    return tf.nn.batch_normalization(
        x = x, mean = l.p['mean'], variance = l.p['var'], 
        offset = None, scale = l.p['scale'], name = name,
        variance_epsilon = 1e-10)

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, l, x):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def dense(l, x, name):
    return tf.nn.xw_plus_b(x, l.p['weights'], 
        l.p['biases'], name = name)
    
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
