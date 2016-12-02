from layer import Layer
import numpy as np

class select_layer(Layer):
    def setup(self, inp, old, 
              out, keep, train,
              activation):
        self.old = old
        self.keep = keep
        self.train = train
        self.activation = activation
        self.inp, self.out = inp, out
        self.wshape = {
            'biases': [out],
            'weights': [inp, out]
        }

    def present(self):
        args = [self.number, 'connected']
        args += [self.inp, self.old]
        args += [self.activation]
        self.presenter = connected_layer(*args)

    def recollect(self, val):
        w = val['weights']
        b = val['biases']
        if w is None:
            self.w = val
            return

        keep_b = np.take(b, self.keep)
        keep_w = np.take(w, self.keep, 1)
        train_b = b[self.train:]
        train_w = w[:, self.train:]
        self.w['biases'] = np.concatenate(
            (keep_b, train_b))
        self.w['weights'] = np.concatenate(
            (keep_w, train_w), axis = 1)


class connected_layer(Layer):
    def setup(self, input_size, 
              output_size, activation):
        self.activation = activation
        self.inp = input_size
        self.out = output_size
        self.wshape = {
            'biases': [self.out],
            'weights': [self.inp, self.out]
        }

    def finalize(self, transpose):
        weights = self.w['weights']
        if weights is None: return
        shp = self.wshape['weights']
        if not transpose:
            weights = weights.reshape(shp[::-1])
            weights = weights.transpose([1,0])
        else: weights = weights.reshape(shp)
        self.w['weights'] = weights