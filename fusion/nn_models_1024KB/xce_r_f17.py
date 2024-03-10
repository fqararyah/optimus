from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
xce_r_f17

"""
batch_size = get_batch_size()
NN = Network('xce_r_f17')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

#layers: 1
NN.add('conv1', ConvLayer(3, 32, 112, 3, 2, nimg=batch_size))

#layers: 2-4
NN.add('conv2', ConvLayer(32, 64, 112, 3, 1, nimg=batch_size))
RES_PREV = 'conv2'

#layers: 3-9
NN.add('conv3_a_dw', DWConvLayer(64, 112, 3, 1, nimg=batch_size))
NN.add('conv3_b_pw', ConvLayer(64, 128, 112, 1, 1, nimg=batch_size))
NN.add('conv3_c_dw', DWConvLayer(128, 112, 3, 1, nimg=batch_size))
NN.add('conv3_d_pw', ConvLayer(128, 128, 112, 1, 1, nimg=batch_size))
NN.add('pool1', PoolingLayer(128, 56, 3, 2, nimg=batch_size))
NN.add('conv3_e_pw', ConvLayer(64, 128, 56, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
NN.add('conv3_res', EltwiseLayer(128, 56, 2, nimg=batch_size),
        prevs=('pool1', 'conv3_e_pw'))
RES_PREV = 'conv3_res'

# #layers: 11-17
NN.add('conv4_a_dw', DWConvLayer(128, 56, 3, 1, nimg=batch_size))
NN.add('conv4_b_pw', ConvLayer(128, 256, 56, 1, 1, nimg=batch_size))
NN.add('conv4_c_dw', DWConvLayer(256, 56, 3, 1, nimg=batch_size))
NN.add('conv4_d_pw', ConvLayer(256, 256, 56, 1, 1, nimg=batch_size))
NN.add('pool2', PoolingLayer(256, 28, 3, 2, nimg=batch_size))
NN.add('conv4_e_pw', ConvLayer(128, 256, 28, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
NN.add('conv4_res', EltwiseLayer(256, 28, 2, nimg=batch_size),
        prevs=('pool2', 'conv4_e_pw'))
RES_PREV = 'conv4_res'

#layers: 19-25
NN.add('conv5_a_dw', DWConvLayer(256, 28, 3, 1, nimg=batch_size))
NN.add('conv5_b_pw', ConvLayer(256, 728, 28, 1, 1, nimg=batch_size))
NN.add('conv5_c_dw', DWConvLayer(728, 28, 3, 1, nimg=batch_size))
NN.add('conv5_d_pw', ConvLayer(728, 728, 28, 1, 1, nimg=batch_size))
NN.add('pool3', PoolingLayer(728, 28, 3, 2, nimg=batch_size))
NN.add('conv5_e_pw', ConvLayer(256, 728, 14, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
NN.add('conv5_res', EltwiseLayer(728, 14, 2, nimg=batch_size),
        prevs=('pool3', 'conv5_e_pw'))