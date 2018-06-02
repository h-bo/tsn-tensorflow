import caffe
import pdb
import json
import numpy as np
from utils import *

model_path = './inception_v3_kinetics_rgb_pretraine/inception_v3_rgb_deploy.prototxt' 
param_path = './inception_v3_kinetics_rgb_pretraine/inception_v3_kinetics_rgb_pretrained.caffemodel' 
caffe.set_mode_cpu
net = caffe.Net(model_path, param_path, caffe.TEST)

save_name = 'tsn_%s_params' %'rgb'
order_name = '%s_names'%save_name
# bn_order_name = 'bn_%s'%order_name
dict_name = net.params
# bn_names_order = [t for t in net.params.keys() if 'batchnorm' in t]
param_dict = {}
for key,value in net.params.items():
    param_dict[key] = [t.data for t in value]

save_obj(param_dict, save_name)
save_obj(net.params.keys(), order_name)
# save_obj(bn_names_order, bn_order_name)

t = load_obj(save_name)  

save_name = 'tsn_%s_params' %'rgb'
order_name = '%s_names'%save_name
param_dict = load_obj(save_name)  
names_order = load_obj(order_name) 

pdb.set_trace()