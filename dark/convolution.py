from layer import Layer

class local_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, w_, h_, activation):
        self.pad = pad * (ksize / 2)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.h_out = h_
        self.w_out = w_

        self.dnshape = [h_ * w_, n, c, ksize, ksize]
        self.wshape = dict({
            'biases': [h_ * w_ * n],
            'kernels': [h_ * w_, ksize, ksize, c, n]
        })

    def finalize(self, _):
        weights = self.w['kernels']
        if weights is None: return
        weights = weights.reshape(self.dnshape)
        weights = weights.transpose([0,3,4,2,1])
        self.w['kernels'] = weights


class convolutional_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.dnshape = [n, c, ksize, ksize] # darknet shape
        self.wshape = dict({
            'biases': [n], 
            'kernel': [ksize, ksize, c, n]
        })
        if self.batch_norm:
            self.wshape.update({
                'moving_variance'  : [n], 
                'moving_mean': [n], 
                'gamma' : [n]
            })
            self.h['is_training'] = {
                'feed': True,
                'dfault': False,
                'shape': ()
            }

    def finalize(self, _):
        """deal with darknet"""
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.dnshape)
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel