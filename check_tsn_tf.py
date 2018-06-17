import tensorflow as tf
import numpy as np
import math
import os
import pdb
import caffe
#from ipdb import set_trace

import sys
from tsn_net import *

test_input = np.array(np.ones((1,299,299,3)), np.float32)
# # with arg_scope(inception_v3_arg_scope()):
# tsn_feat_flat, end_points = tsn_inception_v3(test_input, modality = 'rgb', crop_num=1,)#reuse=(mode=='test')
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run([init_op])
#     tsn_feat_flat_np = sess.run([tsn_feat_flat])
# pdb.set_trace()
model_path = '/DATA/data/bhuang/train/tsn_tf/inception_v3_kinetics_rgb_pretraine/inception_v3_rgb_deploy.prototxt' 
param_path = '/DATA/data/bhuang/train/tsn_tf/inception_v3_kinetics_rgb_pretraine/inception_v3_kinetics_rgb_pretrained.caffemodel' 
caffe.set_mode_cpu
net = caffe.Net(model_path, param_path, caffe.TEST)
# pdb.set_trace()
net.blobs['data'].data[...] = test_input.transpose([0,3,1,2])
out = net.forward()
# pdb.set_trace()