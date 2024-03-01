from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, DWConvLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
ResNet-50

He, Zhang, Ren, and Sun, 2015
"""
batch_size = get_batch_size()
NN = Network('ResNet50')

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
           prevs=(RES_PREV, 'conv4_res'))

#layers: 12-16
NN.add('conv5_a_pw', ConvLayer(144, 24, 56, 1, 1, nimg=batch_size))
NN.add('conv5_b_dw', DWConvLayer(144, 28, 3, 2, nimg=batch_size))
NN.add('conv5_c_pw', ConvLayer(32, 144, 56, 1, 1, nimg=batch_size))
RES_PREV = 'conv5_c_pw'

#layers: 16-19
NN.add('conv6_a_pw', ConvLayer(192, 32, 28, 1, 1, nimg=batch_size))
NN.add('conv6_b_dw', DWConvLayer(192, 28, 3, 1, nimg=batch_size))
NN.add('conv6_c_pw', ConvLayer(32, 192, 28, 1, 1, nimg=batch_size))
NN.add('conv6_res', EltwiseLayer(32, 28, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv6_c_pw'))
RES_PREV = 'conv6_res'

#layers: 20-23
NN.add('conv7_a_pw', ConvLayer(192, 32, 28, 1, 1, nimg=batch_size))
NN.add('conv7_b_dw', DWConvLayer(192, 28, 3, 1, nimg=batch_size))
NN.add('conv7_c_pw', ConvLayer(32, 192, 28, 1, 1, nimg=batch_size))
NN.add('conv7_res', EltwiseLayer(32, 28, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv7_c_pw'))

#layers: 24-28
NN.add('conv8_a_pw', ConvLayer(192, 32, 28, 1, 1, nimg=batch_size))
NN.add('conv8_b_dw', DWConvLayer(192, 14, 3, 2, nimg=batch_size))
NN.add('conv8_c_pw', ConvLayer(64, 144, 14, 1, 1, nimg=batch_size))
RES_PREV = 'conv8_c_pw'

#layers: 28-31
NN.add('conv9_a_pw', ConvLayer(384, 64, 14, 1, 1, nimg=batch_size))
NN.add('conv9_b_dw', DWConvLayer(384, 14, 3, 1, nimg=batch_size))
NN.add('conv9_c_pw', ConvLayer(64, 384, 14, 1, 1, nimg=batch_size))
NN.add('conv9_res', EltwiseLayer(64, 14, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv9_c_pw'))
RES_PREV = 'conv9_res'

#layers: 32-35
NN.add('conv10_a_pw', ConvLayer(384, 64, 14, 1, 1, nimg=batch_size), prevs=(RES_PREV,))
NN.add('conv10_b_dw', DWConvLayer(384, 14, 3, 1, nimg=batch_size))
NN.add('conv10_c_pw', ConvLayer(64, 384, 14, 1, 1, nimg=batch_size))
NN.add('conv10_res', EltwiseLayer(64, 14, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv10_c_pw'))
RES_PREV = 'conv10_res'

#layers: 36-39
NN.add('conv11_a_pw', ConvLayer(384, 64, 14, 1, 1, nimg=batch_size), prevs=(RES_PREV,))
NN.add('conv11_b_dw', DWConvLayer(384, 14, 3, 1, nimg=batch_size))
NN.add('conv11_c_pw', ConvLayer(64, 384, 14, 1, 1, nimg=batch_size))
NN.add('conv11_res', EltwiseLayer(64, 14, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv11_c_pw'))

#layers: 40-43
NN.add('conv12_a_pw', ConvLayer(384, 64, 14, 1, 1, nimg=batch_size))
NN.add('conv12_b_dw', DWConvLayer(384, 14, 3, 1, nimg=batch_size))
NN.add('conv12_c_pw', ConvLayer(96, 384, 14, 1, 1, nimg=batch_size))
RES_PREV = 'conv12_c_pw'

#layers: 43-46
NN.add('conv13_a_pw', ConvLayer(576, 96, 14, 1, 1, nimg=batch_size))
NN.add('conv13_b_dw', DWConvLayer(576, 14, 3, 1, nimg=batch_size))
NN.add('conv13_c_pw', ConvLayer(96, 576, 14, 1, 1, nimg=batch_size))
NN.add('conv13_res', EltwiseLayer(96, 7, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv13_c_pw'))
RES_PREV = 'conv13_res'

#layers: 47-50
NN.add('conv14_a_pw', ConvLayer(576, 96, 14, 1, 1, nimg=batch_size))
NN.add('conv14_b_dw', DWConvLayer(576, 14, 3, 1, nimg=batch_size))
NN.add('conv14_c_pw', ConvLayer(96, 576, 14, 1, 1, nimg=batch_size))
NN.add('conv14_res', EltwiseLayer(96, 7, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv14_c_pw'))

#layers: 51-54
NN.add('conv15_a_pw', ConvLayer(576, 64, 14, 1, 1, nimg=batch_size))
NN.add('conv15_b_dw', DWConvLayer(576, 7, 3, 2, nimg=batch_size))
NN.add('conv15_c_pw', ConvLayer(160, 576, 7, 1, 1, nimg=batch_size))
RES_PREV = 'conv15_c_pw'

#layers: 55-58
NN.add('conv16_a_pw', ConvLayer(960, 160, 7, 1, 1, nimg=batch_size))
NN.add('conv16_b_dw', DWConvLayer(960, 7, 3, 1, nimg=batch_size))
NN.add('conv16_c_pw', ConvLayer(160, 960, 7, 1, 1, nimg=batch_size))
NN.add('conv16_res', EltwiseLayer(160, 7, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv16_c_pw'))
RES_PREV = 'conv16_res'

#layers: 59-62
NN.add('conv17_a_pw', ConvLayer(960, 160, 7, 1, 1, nimg=batch_size))
NN.add('conv17_b_dw', DWConvLayer(960, 7, 3, 1, nimg=batch_size))
NN.add('conv17_c_pw', ConvLayer(160, 960, 7, 1, 1, nimg=batch_size))
NN.add('conv17_res', EltwiseLayer(160, 7, 2, nimg=batch_size),
           prevs=(RES_PREV, 'conv17_c_pw'))

#layers: 63-65
NN.add('conv18_a_pw', ConvLayer(960, 160, 7, 1, 1, nimg=batch_size))
NN.add('conv18_b_dw', DWConvLayer(960, 7, 3, 1, nimg=batch_size))
NN.add('conv18_c_pw', ConvLayer(320, 960, 7, 1, 1, nimg=batch_size))

#layers: 66
NN.add('conv19_a_pw', ConvLayer(1280, 320, 7, 1, 1, nimg=batch_size))

NN.add('pool19', PoolingLayer(1280, 1, 7, nimg=batch_size))

NN.add('fc', FCLayer(1280, 1000, nimg=batch_size))
