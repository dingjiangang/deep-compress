import numpy as np
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

################################################################################
####################### DATA ###################################################
################################################################################
def unpickle(file):
	with open(file, 'rb') as fo:
		dict_data = pickle.load(fo, encoding='bytes')
	if b'data' in dict_data:
		dict_data[b'data'] = dict_data[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2) / 256.
	return dict_data

def load_data_one(f):
	batch = unpickle(f)
	data = batch[b'data']
	labels = batch[b'labels']
	print("Loading %s: %d" % (f, len(data)))
	return data, labels

def load_data(files, data_dir, n_classes):
	data, labels = load_data_one(data_dir + '/' + files[0])
	for f in files[1:]:
		data_n, labels_n = load_data_one(data_dir + '/' + f)
		data = np.append(data, data_n, axis=0)
		labels = np.append(labels, labels_n, axis=0)
	labels = np.array([ [ float(i == label) for i in range(n_classes) ] for label in labels ])
	return data, labels

def shuffle_data(X_train, y_train):
	X_train, y_train = shuffle(X_train, y_train, random_state=0)
	return X_train, y_train

data_dir = './data'
data_file = data_dir + '/batches.meta'

image_size = 32
image_dim = [image_size , image_size , 3]
meta = unpickle(data_file)
label_names = meta[b'label_names']
n_classes = len(label_names)

train_files = [ 'data_batch_%d' % d for d in range(1, 6) ]
train_data, train_labels = load_data(train_files, data_dir, n_classes)

train_data, train_labels = shuffle(train_data, train_labels)
X_train, X_validation, y_train, y_validation = \
		train_test_split(train_data, train_labels, test_size=0.1)
X_test, y_test = load_data([ 'test_batch' ], data_dir, n_classes)

print("Train:", np.shape(X_train), np.shape(y_train))
print("Validation: ", np.shape(X_validation), np.shape(y_validation))
print("Test:", np.shape(X_test), np.shape(y_test))
data = {'train-data': X_train,
		'train-labels': y_train,
		'vaidation-data': X_validation,
		'validation-labels': y_validation,
		'test-data': X_test,
		'test-labels': y_test }

################################################################################
####################### MODEL ##################################################
################################################################################



def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):															
	res = [ 0 ] * len(tensors)
	batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]										
	total_size = len(batch_tensors[0][1])																																								
	batch_count = (total_size + batch_size - 1) / batch_size
	for batch_idx in range(batch_count):
		current_batch_size = None																																													
		for (placeholder, tensor) in batch_tensors:
			batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]
			current_batch_size = len(batch_tensor)
			feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]
	tmp = session.run(tensors, feed_dict=feed_dict)																																		
	res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]																									 
	return [ r / float(total_size) for r in res ]

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
	W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
	conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
	if with_bias:
		return conv + bias_variable([ out_features ])
	return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
	current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
	current = tf.nn.relu(current)
	current = conv2d(current, in_features, out_features, kernel_size)
	current = tf.nn.dropout(current, keep_prob)
	return current

def block(input, layers, in_features, growth, is_training, keep_prob):
	current = input
	features = in_features
	for idx in range(layers):
		tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
		current = tf.concat(axis=3, values=(current, tmp))
		features += growth
	return current, features

def avg_pool(input, s):
	return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

#def run_model(data, image_dim, n_classes, depth):
depth = 40
growth = 12
weight_decay = 1e-4
layers = (depth - 4) // 3

x = tf.placeholder("float", shape=[None,32,32,3])
y = tf.placeholder("float", shape=[None,n_classes])
lr = tf.placeholder("float", shape=[])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder("bool", shape=[])
current = conv2d(x, 3, 16, 3)

current, features = block(current, layers, 16, 12, is_training, keep_prob)
current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
current = avg_pool(current, 2)
current, features = block(current, layers, features, 12, is_training, keep_prob)
current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
current = avg_pool(current, 2)
current, features = block(current, layers, features, 12, is_training, keep_prob)

current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
current = tf.nn.relu(current)
current = avg_pool(current, 8)
final_dim = features
current = tf.reshape(current, [ -1, final_dim ])
Wfc = weight_variable([ final_dim, n_classes ])
bfc = bias_variable([ n_classes ])
y_ = tf.nn.softmax( tf.matmul(current, Wfc) + bfc )

cross_entropy = -tf.reduce_mean(y * tf.log(y_ + 1e-12))
l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as session:
	batch_size = 64
	learning_rate = 0.1
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	train_data, train_labels = data['train-data'], data['train-labels']
	batch_count = len(train_data) // batch_size
	batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
	batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
	print("Batch per epoch: ", batch_count)
	for epoch in range(1, 1+300):
		if epoch == 150:
			learning_rate = 0.01
		if epoch == 225:
			learning_rate = 0.001
		for batch_idx in range(batch_count):
			x_, y_ = batches_data[batch_idx], batches_labels[batch_idx]
			batch_res = session.run([ train_step, cross_entropy, accuracy ],
				feed_dict = { 	x: x_, 
								y: y_, 
								lr: learning_rate, 
								is_training: True, 
								keep_prob: 0.8 })
			if batch_idx % 100 == 0: 
				print(epoch, batch_idx, batch_res[1:])

		save_path = saver.save(session, 'densenet_%d.ckpt' % epoch)
		test_results = run_in_batch_avg(session,[cross_entropy,accuracy],[x,y],
				feed_dict = { 	x: data['test-data'], 
								y: data['test-labels'], 
								is_training: False, 
								keep_prob: 1. })
		print(epoch, batch_res[1:], test_results)

# #def run():

# run_model(data, image_dim, n_classes, 40)

#run()


# import tensorflow as tf

# import input_MNIST_data
# from input_MNIST_data import shuffle_data
# data = input_MNIST_data.read_data_sets("./data/", one_hot=True)

# import numpy as np
# import sys
# import dill
# import collections

# import pickle

# import pandas as pd

# from sklearn.cluster import KMeans
# from numpy import linalg as LA

# print('----------------------------------------------')
# print('architecture: LeNet-5 --- Data Set: MNIST')
# print('----------------------------------------------')

# # input and output shape
# n_input	 = data.train.images.shape[1]	# here MNIST data input (28,28)
# n_classes = data.train.labels.shape[1]	# here MNIST (0-9 digits)

# # dropout rate
# dropout_rate = 0.5
# # number of weights and bias in each layer
# n_W = {}
# n_b = {}

# # network architecture hyper parameters
# input_shape = [-1,28,28,1]
# W0 = 28
# H0 = 28

# # Layer 1 -- conv
# D1 = 1
# F1 = 5
# K1 = 20
# S1 = 1
# W1 = (W0 - F1) // S1 + 1
# H1 = (H0 - F1) // S1 + 1
# conv1_dim = [F1, F1, D1, K1]
# conv1_strides = [1,S1,S1,1] 
# n_W['conv1'] = F1 * F1 * D1 * K1
# n_b['conv1'] = K1 

# # Layer 2 -- max pool
# D2 = K1
# F2 = 2
# K2 = D2
# S2 = 2
# W2 = (W1 - F2) // S2 + 1
# H2 = (H1 - F2) // S2 + 1
# layer2_ksize = [1,F2,F2,1]
# layer2_strides = [1,S2,S2,1]

# # Layer 3 -- conv
# D3 = K2
# F3 = 5
# K3 = 50
# S3 = 1
# W3 = (W2 - F3) // S3 + 1
# H3 = (H2 - F3) // S3 + 1
# conv2_dim = [F3, F3, D3, K3]
# conv2_strides = [1,S3,S3,1] 
# n_W['conv2'] = F3 * F3 * D3 * K3
# n_b['conv2'] = K3 

# # Layer 4 -- max pool
# D4 = K3
# F4 = 2
# K4 = D4
# S4 = 2
# W4 = (W3 - F4) // S4 + 1
# H4 = (H3 - F4) // S4 + 1
# layer4_ksize = [1,F4,F4,1]
# layer4_strides = [1,S4,S4,1]


# # Layer 5 -- fully connected
# n_in_fc = W4 * H4 * D4
# n_hidden = 500
# fc_dim = [n_in_fc,n_hidden]
# n_W['fc'] = n_in_fc * n_hidden
# n_b['fc'] = n_hidden

# # Layer 6 -- output
# n_in_out = n_hidden
# n_W['out'] = n_hidden * n_classes
# n_b['out'] = n_classes

# for key, value in n_W.items():
# 	n_W[key] = int(value)

# for key, value in n_b.items():
# 	n_b[key] = int(value)

# # tf Graph input
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])

# learning_rate = tf.placeholder("float")
# momentum_tf = tf.placeholder("float")
# mu_tf = tf.placeholder("float")

# # weights of LeNet-5 CNN -- tf tensors
# weights = {
#		 # 5 x 5 convolution, 1 input image, 20 outputs
#		 'conv1': tf.get_variable('w_conv1', shape=[F1, F1, D1, K1],
#									initializer=tf.contrib.layers.xavier_initializer()),
#		 # 'conv1': tf.Variable(tf.random_normal([F1, F1, D1, K1])),
#		 # 5x5 conv, 20 inputs, 50 outputs 
#		 #'conv2': tf.Variable(tf.random_normal([F3, F3, D3, K3])),
#		 'conv2': tf.get_variable('w_conv2', shape=[F3, F3, D3, K3],
#									initializer=tf.contrib.layers.xavier_initializer()),
#		 # fully connected, 800 inputs, 500 outputs
#		 #'fc': tf.Variable(tf.random_normal([n_in_fc, n_hidden])),
#		 'fc': tf.get_variable('w_fc', shape=[n_in_fc, n_hidden],
#									initializer=tf.contrib.layers.xavier_initializer()),
#		 # 500 inputs, 10 outputs (class prediction)
#		 #'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
#		 'out': tf.get_variable('w_out', shape=[n_hidden, n_classes],
#									initializer=tf.contrib.layers.xavier_initializer())
# }

# # biases of LeNet-5 CNN -- tf tensors
# biases = {
#		 'conv1': tf.get_variable('b_conv1', shape=[K1],
#									initializer=tf.zeros_initializer()),
#		 'conv2': tf.get_variable('b_conv2', shape=[K3],
#									initializer=tf.zeros_initializer()),
#		 'fc': tf.get_variable('b_fc', shape=[n_hidden],
#									initializer=tf.zeros_initializer()),
#		 'out': tf.get_variable('b_out', shape=[n_classes],
#									initializer=tf.zeros_initializer()) 
#		 # 'conv1': tf.Variable(tf.random_normal([K1])),
#		 # 'conv2': tf.Variable(tf.random_normal([K3])),
#		 # 'fc': tf.Variable(tf.random_normal([n_hidden])),
#		 # 'out': tf.Variable(tf.random_normal([n_classes]))
# }

# def model(x,_W,_b):
# 	# Reshape input to a 4D tensor 
#		 x = tf.reshape(x, shape = input_shape)
#		 # LAYER 1 -- Convolution Layer
#		 conv1 = tf.nn.relu(tf.nn.conv2d(input = x, 
#		 								filter =_W['conv1'],
#		 								strides = [1,S1,S1,1],
#		 								padding = 'VALID') + _b['conv1'])
#		 # Layer 2 -- max pool
#		 conv1 = tf.nn.max_pool(	value = conv1, 
#		 						ksize = [1, F2, F2, 1], 
#		 						strides = [1, S2, S2, 1], 
#		 						padding = 'VALID')

#		 # LAYER 3 -- Convolution Layer
#		 conv2 = tf.nn.relu(tf.nn.conv2d(input = conv1, 
#		 								filter =_W['conv2'],
#		 								strides = [1,S3,S3,1],
#		 								padding = 'VALID') + _b['conv2'])
#		 # Layer 4 -- max pool
#		 conv2 = tf.nn.max_pool(	value = conv2 , 
#		 						ksize = [1, F4, F4, 1], 
#		 						strides = [1, S4, S4, 1], 
#		 						padding = 'VALID')
#		 # Fully connected layer
#		 # Reshape conv2 output to fit fully connected layer
#		 fc = tf.contrib.layers.flatten(conv2)
#		 fc = tf.nn.relu(tf.matmul(fc, _W['fc']) + _b['fc'])
#		 fc = tf.nn.dropout(fc, dropout_rate)

#		 output = tf.matmul(fc, _W['out']) + _b['out']
#		 # output = tf.nn.dropout(output, keep_prob = dropout_rate)
#		 return output

# # Construct model
# output = model(x,weights,biases)
# # Softmax loss
# loss = tf.reduce_mean(
# 	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))
# correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # REFERENCE MODEL Parameters -- for training the Reference model: 

# # Batch size
# minibatch = 512
# # Total minibatches
# total_minibatches = 100000
# # number of minibatches in data
# num_minibatches_data = data.train.images.shape[0] // minibatch

# # Learning rate
# lr = 0.02
# # Learning rate decay:	every 2000 minibatches
# learning_rate_decay = 0.99
# learning_rate_stay_fixed = 1000

# # Optimizer: Nesterov accelerated gradient with momentum 0.95
# # this is for training the reference net
# momentum = 0.9

# optimizer = tf.train.MomentumOptimizer(
# 	learning_rate = learning_rate,
# 	momentum = momentum_tf,
# 	use_locking=False,
# 	name='Momentum',
# 	use_nesterov=True)

# GATE_NONE = 0
# GATE_OP = 1
# GATE_GRAPH = 2
# # GATE_OP:
# # For each Op, make sure all gradients are computed before
# # they are used.	This prevents race conditions for Ops that generate gradients
# # for multiple inputs where the gradients depend on the inputs.
# train = optimizer.minimize(
#		 loss,
#		 global_step=None,
#		 var_list=None,
#		 gate_gradients=GATE_OP,
#		 aggregation_method=None,
#		 colocate_gradients_with_ops=False,
#		 name='train',
#		 grad_loss=None)


# saver = tf.train.Saver()

# init = tf.global_variables_initializer()

# ###############################################################################
# ######## training data and neural net architecture with weights w #############
# ###############################################################################
# print('----------------------------------------------')
# print('TRAINGING REFERENCE NET for LeNet-5')
# print('----------------------------------------------')
# ################### TO SAVE TRAINING AND TEST LOSS AND ERROR ##################
# ################### FOR REFERENCE NET #########################################
# num_epoch_ref = total_minibatches // num_minibatches_data
# epoch_ref_vec = np.array(range(num_epoch_ref+1)) 
# train_loss_ref = np.zeros(num_epoch_ref+1)
# train_error_ref = np.zeros(num_epoch_ref+1)
# val_loss_ref = np.zeros(num_epoch_ref+1)
# val_error_ref = np.zeros(num_epoch_ref+1)
# test_loss_ref = np.zeros(num_epoch_ref+1)
# test_error_ref = np.zeros(num_epoch_ref+1)

# ################### TO SAVE MODEL ##################
# model_file_name = 'reference_model_lenet_5.ckpt'
# model_file_path = './model_lenet_5/' + model_file_name 

# ############################## TRAIN LOOP #####################################
# with tf.Session() as sess:
# 	sess.run(init)
# 	for i in range(total_minibatches):
# 		index_minibatch = i % num_minibatches_data
# 		epoch = i // num_minibatches_data		
# 		# shuffle data at the begining of each epoch
# 		if index_minibatch == 0:
# 			X_train, y_train = shuffle_data(data)
# 		# adjust learning rate
# 		if i % learning_rate_stay_fixed == 0:
# 			j = i // learning_rate_stay_fixed
# 			lr = 0.02 * 0.99 ** j
# 		# mini batch 
# 		start_index = index_minibatch		 * minibatch
# 		end_index	 = (index_minibatch+1) * minibatch
# 		X_batch = X_train[start_index:end_index]
# 		y_batch = y_train[start_index:end_index]

# 		train.run(feed_dict = { x: X_batch,
# 					 			y: y_batch,
# 								learning_rate: lr,
# 								momentum_tf: momentum})
		
# 		############### LOSS AND ACCURACY EVALUATION ##########################
# 		if index_minibatch == 0:
# 			train_loss, train_accuracy = \
# 					sess.run([loss, accuracy], feed_dict = {x: X_batch, 
# 																y: y_batch} )
# 			train_loss_ref[epoch] = train_loss
# 			train_error_ref[epoch] = 1 - train_accuracy

# 			val_loss, val_accuracy = \
# 			sess.run([loss, accuracy], feed_dict = {x: data.validation.images, 
# 													y: data.validation.labels} )
# 			val_loss_ref[epoch] = val_loss
# 			val_error_ref[epoch] = 1 - val_accuracy

# 			test_loss, test_accuracy = \
# 			sess.run([loss, accuracy], feed_dict = {x: data.test.images, 
# 													y: data.test.labels} )
# 			test_loss_ref[epoch] = test_loss
# 			test_error_ref[epoch] = 1 - test_accuracy

# 			print('step: {}, train loss: {}, train acuracy: {}' \
# 				.format(i, train_loss, train_accuracy) )
# 			print('step: {}, val loss: {}, val acuracy: {}' \
# 				.format(i, val_loss, val_accuracy) )
# 			print('step: {}, test loss: {}, test acuracy: {}' \
# 				.format(i, test_loss, test_accuracy) )
# 		#train_loss_ref = sess.run(loss)
		
# 	save_path = saver.save(sess, model_file_path)
# 	# reference weight and bias
# 	w_bar = sess.run(weights)
# 	bias_bar = sess.run(biases)


# df_ref = pd.DataFrame({	'train_loss_ref' : train_loss_ref,
# 						'train_error_ref': train_error_ref,
# 						'val_loss_ref': val_loss_ref,
# 						'val_error_ref': val_error_ref,
# 						'test_loss_ref': test_loss_ref,
# 						'test_error_ref': test_error_ref})


# file_pickle = './results_lenet_5/df_ref_lenet_5_pickle.pkl'
# with open(file_pickle,'wb') as f:
# 	df_ref.to_pickle(f)

# weights_pickle = './results_lenet_5/weights_biases_lenet_5_ref_pickle.pkl'

# with open(weights_pickle,'wb') as f:
# 	pickle.dump(w_bar,f,protocol=pickle.HIGHEST_PROTOCOL)
# 	pickle.dump(bias_bar,f,protocol=pickle.HIGHEST_PROTOCOL)

# # with tf.Session() as sess:
# #		 saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
# #		 saver.restore(sess, "/tmp/model.ckpt")
# # reference weight and bias
# 	# w_bar = sess.run(weights)
# 	# bias_bar = sess.run(biases)