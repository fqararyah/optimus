from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
xce_r_l67

"""
batch_size = get_batch_size()
NN = Network('xce_r_l67')

NN.set_input_layer(InputLayer(728, 14, nimg=batch_size))

current_bott_index = 6
repeat = 8
#layers: 27-33
RES_PREV =''
for bott_index in range(current_bott_index, current_bott_index + repeat):
    NN.add('conv{}_a_dw'.format(bott_index), DWConvLayer(728, 14, 3, 1, nimg=batch_size))
    if bott_index == current_bott_index:
        RES_PREV = 'conv{}_a_dw'.format(bott_index)
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
