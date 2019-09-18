import tensorflow as tf
import tf_utils as util
import numpy as np


def get_model(inputs,is_training,bn_decay = None):
	
	stem = util.conv2d(inputs,32,[3,3],scope='conv1',stride = [2,2],padding = 'VALID',use_xavier = True,is_training = is_training,bn_decay=bn_decacy)
	stem = util.conv2d(stem,32,[3,3],scope='conv2',stride = [1,1],padding = 'VALID',use_xavier = True,is_training = is_training,bn_decay=bn_decacy)
	stem = util.conv2d(stem,64,[3,3],scope='conv3',stride = [1,1],padding = 'VALID',use_xavier = True,is_training = is_training,bn_decay=bn_decacy)
	
	

	
	
	
"""
	layer1 = util.conv2d(inputs,32,[3,3],'stem_1',stride = [2,2],padding = 'VALID',use_xavier = True)
	layer1 = util.conv2d(layer1,32,[3,3],'stem_1',stride = [1,1],padding = 'VALID',use_xavier = True)
	layer1 = util.conv2d(layer1,64,[3,3],'stem_1',stride = [1,1],padding = 'SAME',use_xavier = True)
	
	layer2_1 = util.conv2d(layer1,96,[3,3],'stem_2',stride = [2,2],padding = 'VALID',use_xavier = True)
	layer2_2 = util.max_pool2d(layer1,[3,3],'stem_2',stride = [2,2],padding = 'VALID')
	
	layer2_concat = util.filter_concat(3,[layer2_1,layer2_2])
	
	layer3_1_1 = util.conv2d(layer2_concat,64,[1,1],'stem_3',stride = [1,1],padding = 'SAME',use_xavier = True)
	layer3_1_2 = util.conv2d(layer3_1_1,96,[3,3],'stem_3',stride = [1,1],padding = 'VALID',use_xavier=True)
	
	layer3_2_1 = util.conv2d(layer2_concat,64,[1,1],'stem_3',stride = [1,1],padding = 'SAME',use_xavier = True)
	
"""
	
	
	
	