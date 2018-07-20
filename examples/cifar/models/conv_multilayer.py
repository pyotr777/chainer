from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L


# Number of layers and their output channels 
# are set with out_channels parameter.
class conv_multilayer(chainer.Chain):

    def __init__(self, out_channels=[64], class_labels=10):
        super(conv_multilayer, self).__init__()
        with self.init_scope():
            conv_layers = []
            for conv in out_channels:
                conv_layers.append( L.Convolution2D(None, conv, 3, pad=1, nobias=True))
            self.convolution_layers = chainer.ChainList(*conv_layers)
            self.fc = L.Linear(None, class_labels, nobias=True)
        print("Using {} Convolution2D layers with {} output channels.".format(len(conv_layers),out_channels))

    def __call__(self, x):
        conv_layers = self.convolution_layers
        h = conv_layers[0](x)
        for layer in conv_layers[1:]:
            h = layer(h)
            h = F.relu(h)
        return self.fc(h)
