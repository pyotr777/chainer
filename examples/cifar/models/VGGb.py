from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import cupy as cp

class Block(chainer.Chain):


    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGGb(chainer.Chain):

    def __init__(self, class_labels=10):
        super(VGGb, self).__init__()
        with self.init_scope():
            self.block5_1 = Block(512, 3)
            self.block5_2 = Block(512, 3)
            self.block5_3 = Block(512, 3)

            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        # print("x shape: {}".format(x.shape)) # x shape: (40, 3, 32, 32)
        h = cp.reshape(x,(x.shape[0],512,3,2))
        # 512 channel blocks:
        h = self.block5_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        return self.fc2(h)
