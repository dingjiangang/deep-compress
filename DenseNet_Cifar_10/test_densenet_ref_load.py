import tensorflow as tf
tf.reset_default_graph()

import numpy as np

###############################################################################
######## training data and neural net architecture with weights w #############
###############################################################################
print('----------------------------------------------')
print('LOADING MY PRETRAINED REFERENCE NET for DenseNet-40')
print('----------------------------------------------')
################### TO LOAD MODEL #############################################
model_file_path = './densenet_50.ckpt'
model_file_meta = './densenet_50.ckpt.meta'
############################## LOAD weights and biases ########################
variables = {}
ref_values = {}
with tf.Session() as sess:
	saver = tf.train.import_meta_graph(model_file_meta)
	saver.restore(sess, model_file_path)
	for v in tf.trainable_variables():
		variables[v.name] = v
		ref_values[v.name] = sess.run(v)
