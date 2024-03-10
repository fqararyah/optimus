from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer
from fusion.scheduling.batch_size import get_batch_size

"""
MobileNet_f9

Andrew G. Howard, Menglong Zhu, Bo Chen, ect., 2017
"""
batch_size = get_batch_size()

NN = Network('MobileNet_f9')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))
NN.add('conv1', ConvLayer(3, 32, 112, 3, 2, nimg=batch_size))

NN.add('conv2_1_dw', DWConvLayer(32, 112, 3, 1, nimg=batch_size))
NN.add('conv2_1_pw', ConvLayer(32, 64, 112, 1, 1, nimg=batch_size))

NN.add('conv2_2_dw', DWConvLayer(64, 56, 3, 2, nimg=batch_size))
NN.add('conv2_2_pw', ConvLayer(64, 128, 56, 1, 1, nimg=batch_size))

NN.add('conv3_1_dw', DWConvLayer(128, 56, 3, 1, nimg=batch_size))
NN.add('conv3_1_pw', ConvLayer(128, 128, 56, 1, 1, nimg=batch_size))

NN.add('conv3_2_dw', DWConvLayer(128, 28, 3, 2, nimg=batch_size))
NN.add('conv3_2_pw', ConvLayer(128, 256, 28, 1, 1, nimg=batch_size))

