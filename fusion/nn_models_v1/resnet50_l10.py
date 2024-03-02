from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
ResNet-50

He, Zhang, Ren, and Sun, 2015
"""
batch_size = get_batch_size()
NN = Network('ResNet50_l10')

NN.set_input_layer(InputLayer(1024, 14, nimg=batch_size))
NN.add('conv1', ConvLayer(1024, 1024, 14, 3, 1, nimg=batch_size))
RES_PREV = 'conv1'

for i in range(3):
    NN.add('conv5_{}_a'.format(i),
           ConvLayer(1024, 512, 7, 1, 2, nimg=batch_size) if i == 0
           else ConvLayer(2048, 512, 7, 1, nimg=batch_size))
    NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 7, 3, nimg=batch_size))
    NN.add('conv5_{}_c'.format(i), ConvLayer(512, 2048, 7, 1, nimg=batch_size))

    # With residual shortcut.
    if i == 0:
        NN.add('conv5_br', ConvLayer(1024, 2048, 7, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
        RES_PREV = 'conv5_br'
    NN.add('conv5_{}_res'.format(i), EltwiseLayer(2048, 7, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv5_{}_c'.format(i)))
    RES_PREV = 'conv5_{}_res'.format(i)

NN.add('pool5', PoolingLayer(2048, 1, 7, nimg=batch_size))

NN.add('fc', FCLayer(2048, 1000, nimg=batch_size))
