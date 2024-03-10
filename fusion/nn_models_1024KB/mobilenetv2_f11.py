from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
MobileNetV2_f11

"""
batch_size = get_batch_size()
NN = Network('MobileNetV2_f11')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

#layers: 1
NN.add('conv1', ConvLayer(3, 32, 112, 3, 2, nimg=batch_size))

#layers: 2-4
NN.add('conv2_a_dw', DWConvLayer(32, 112, 3, 1, nimg=batch_size))
NN.add('conv2_b_pw', ConvLayer(32, 16, 112, 1, 1, nimg=batch_size))

#layers: 4-8
NN.add('conv3_a_pw', ConvLayer(16, 96, 112, 1, 1, nimg=batch_size))
NN.add('conv3_b_dw', DWConvLayer(96, 56, 3, 2, nimg=batch_size))
NN.add('conv3_c_pw', ConvLayer(24, 96, 56, 1, 1, nimg=batch_size))
RES_PREV = 'conv3_c_pw'

#layers: 8-11
NN.add('conv4_a_pw', ConvLayer(144, 24, 56, 1, 1, nimg=batch_size))
NN.add('conv4_b_dw', DWConvLayer(144, 56, 3, 1, nimg=batch_size))
NN.add('conv4_c_pw', ConvLayer(24, 144, 56, 1, 1, nimg=batch_size))
NN.add('conv4_res', EltwiseLayer(24, 56, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv4_c_pw'))

#layers: 12-16
NN.add('conv5_a_pw', ConvLayer(144, 24, 56, 1, 1, nimg=batch_size))
NN.add('conv5_b_dw', DWConvLayer(144, 28, 3, 2, nimg=batch_size))