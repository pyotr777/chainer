from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L


class conv(chainer.Chain):


    def __init__(self, out_channels=64, class_labels=10):
        super(conv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, 3, pad=1, nobias=True)
            self.fc = L.Linear(None, class_labels, nobias=True)
        print("Using Convolution2D with {} output channels.".format(out_channels))

    def __call__(self, x):
        h = self.conv(x)
        h = F.relu(h)
        return self.fc(h)
