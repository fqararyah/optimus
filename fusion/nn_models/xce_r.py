from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
xce_r

"""
batch_size = get_batch_size()
NN = Network('xce_r')

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
RES_PREV = 'conv5_res'

current_bott_index = 6
repeat = 8
#layers: 27-33
for bott_index in range(current_bott_index, current_bott_index + repeat):
    NN.add('conv{}_a_dw'.format(bott_index), DWConvLayer(728, 14, 3, 1, nimg=batch_size))
    NN.add('conv{}_b_pw'.format(bott_index), ConvLayer(728, 728, 14, 1, 1, nimg=batch_size))
    NN.add('conv{}_c_dw'.format(bott_index), DWConvLayer(728, 14, 3, 1, nimg=batch_size))
    NN.add('conv{}_d_pw'.format(bott_index), ConvLayer(728, 728, 14, 1, 1, nimg=batch_size))
    NN.add('conv{}_e_dw'.format(bott_index), DWConvLayer(728, 14, 3, 1, nimg=batch_size))
    NN.add('conv{}_f_pw'.format(bott_index), ConvLayer(728, 728, 14, 1, 1, nimg=batch_size))
    NN.add('conv{}_res'.format(bott_index), EltwiseLayer(728, 14, 2, nimg=batch_size),
            prevs=(RES_PREV, 'conv{}_f_pw'.format(bott_index)))
    RES_PREV = 'conv{}_res'.format(bott_index)
current_bott_index += repeat

#layers: 91-97
NN.add('conv{}_a_dw'.format(current_bott_index), DWConvLayer(728, 14, 3, 1, nimg=batch_size))
NN.add('conv{}_b_pw'.format(current_bott_index), ConvLayer(728, 728, 14, 1, 1, nimg=batch_size))
NN.add('conv{}_c_dw'.format(current_bott_index), DWConvLayer(728, 14, 3, 1, nimg=batch_size))
NN.add('conv{}_d_pw'.format(current_bott_index), ConvLayer(728, 728, 7, 1, 1, nimg=batch_size))
NN.add('pool4', PoolingLayer(728, 28, 3, 2, nimg=batch_size))
NN.add('conv{}_e_pw'.format(current_bott_index), ConvLayer(728, 728, 28, 1, 2, nimg=batch_size), prevs=(RES_PREV,))
NN.add('conv{}_res'.format(current_bott_index), EltwiseLayer(728, 7, 2, nimg=batch_size),
        prevs=('pool4', 'conv{}_e_pw'.format(current_bott_index)))
current_bott_index += 1

#layers: 63-65
NN.add('conv{}_a_dw'.format(current_bott_index), DWConvLayer(728, 7, 3, 1, nimg=batch_size))
NN.add('conv{}_b_pw'.format(current_bott_index), ConvLayer(728, 1536, 7, 1, 1, nimg=batch_size))
NN.add('conv{}_c_dw'.format(current_bott_index), DWConvLayer(1536, 7, 3, 1, nimg=batch_size))
NN.add('conv{}_d_pw'.format(current_bott_index), ConvLayer(1536, 2048, 7, 1, 1, nimg=batch_size))

NN.add('pool5', PoolingLayer(2048, 1, 7, nimg=batch_size))

NN.add('fc', FCLayer(2048, 1000, nimg=batch_size))
