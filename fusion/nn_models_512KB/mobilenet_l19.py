from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer
from fusion.scheduling.batch_size import get_batch_size

"""
MobileNet_l19

Andrew G. Howard, Menglong Zhu, Bo Chen, ect., 2017
"""
batch_size = get_batch_size()

NN = Network('MobileNet_l19')

NN.set_input_layer(InputLayer(256, 28, nimg=batch_size))

NN.add('conv4_1_dw', DWConvLayer(256, 28, 3, 1, nimg=batch_size))
NN.add('conv4_1_pw', ConvLayer(256, 256, 28, 1, 1, nimg=batch_size))

NN.add('conv4_2_dw', DWConvLayer(256, 14, 3, 2, nimg=batch_size))
NN.add('conv4_2_pw', ConvLayer(256, 512, 14, 1, 1, nimg=batch_size))

for i in range(1, 6):
    NN.add('conv5_{}_dw'.format(i), DWConvLayer(512, 14, 3, 1, nimg=batch_size))
    NN.add('conv5_{}_pw'.format(i), ConvLayer(512, 512, 14, 1, 1, nimg=batch_size))

NN.add('conv5_6_dw', DWConvLayer(512, 7, 3, 2, nimg=batch_size))
NN.add('conv5_6_pw', ConvLayer(512, 1024, 7, 1, 1, nimg=batch_size))

NN.add('conv6_dw', DWConvLayer(1024, 7, 3, 1, nimg=batch_size))
NN.add('conv6_pw', ConvLayer(1024, 1024, 7, 1, 1, nimg=batch_size))

NN.add('pool6', PoolingLayer(1024, 1, 7, nimg=batch_size))
NN.add('fc', FCLayer(1024, 1000, nimg=batch_size))

