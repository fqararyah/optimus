from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
ResNet-50_f24

He, Zhang, Ren, and Sun, 2015
"""
batch_size = get_batch_size()
NN = Network('ResNet50_f24')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2, nimg=batch_size))
NN.add('pool1', PoolingLayer(64, 56, 3, 2, nimg=batch_size))

RES_PREV = 'pool1'

for i in range(3):
    NN.add('conv2_{}_a'.format(i), ConvLayer(64 if i == 0 else 256, 64, 56, 1, nimg=batch_size))
    NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 56, 3, nimg=batch_size))
    NN.add('conv2_{}_c'.format(i), ConvLayer(64, 256, 56, 1, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv2_br', ConvLayer(64, 256, 56, 1, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv2_br'
    NN.add('conv2_{}_res'.format(i), EltwiseLayer(256, 56, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv2_{}_c'.format(i)))
    RES_PREV = 'conv2_{}_res'.format(i)

for i in range(4):
    NN.add('conv3_{}_a'.format(i),
           ConvLayer(256, 128, 28, 1, 2, nimg=batch_size) if i == 0
           else ConvLayer(512, 128, 28, 1, nimg=batch_size))
    NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 28, 3, nimg=batch_size))
    NN.add('conv3_{}_c'.format(i), ConvLayer(128, 512, 28, 1, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv3_br', ConvLayer(256, 512, 28, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv3_br'
    NN.add('conv3_{}_res'.format(i), EltwiseLayer(512, 28, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv3_{}_c'.format(i)))
    RES_PREV = 'conv3_{}_res'.format(i)
