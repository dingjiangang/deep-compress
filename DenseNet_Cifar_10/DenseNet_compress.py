# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import tensorflow as tf
###############################################################################
######## training data and neural net architecture with weights w #############
###############################################################################
print('---------------------------------------------------')
print('LOADING MY PRETRAINED REFERENCE NET for DenseNet-40')
print('---------------------------------------------------')
################### TO LOAD MODEL #############################################
model_file_path = './densenet_300.ckpt'
model_file_meta = './densenet_300.ckpt.meta'
############################## LOAD weights and biases ########################
ref_variables = {}
ref_values = {}
ref_weights = {}
ref_weights_values = {}
with tf.Session() as sess:
	saver = tf.train.import_meta_graph(model_file_meta)
	saver.restore(sess, model_file_path)
	for v in tf.trainable_variables():
		ref_variables[v.name] = v
		ref_values[v.name] = sess.run(v)
		if 'Batch' not in v.name:
			ref_weights[v.name] = v
			ref_weights_values[v.name] = sess.run(v)

n_W = {}
for layer, _ in ref_weights_values.items():
	n_W[layer] = ref_weights_values[layer].size

tf.reset_default_graph()
import numpy as np
import sys
import collections
import pickle
import pandas as pd
import argparse
# Set up argument parser
ap = argparse.ArgumentParser()
# Single positional argument, nargs makes it optional
ap.add_argument('k', nargs='?', default=2)
# codebook size
k = int(ap.parse_args().k)


from sklearn.cluster import KMeans
from numpy import linalg as LA

print('----------------------------------------------')
print('architecture: DenseNet --- Data Set: Cifar 10')
print('----------------------------------------------')

print('----------------------------------------------')
print('Compression Algorithm for k = {}' .format(k))
print('----------------------------------------------')

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
		'validation-data': X_validation,
		'validation-labels': y_validation,
		'test-data': X_test,
		'test-labels': y_test }

################################################################################
####################### MODEL ##################################################
################################################################################

def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=64):															
	res = [ 0 ] * len(tensors)
	batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]										
	total_size = len(batch_tensors[0][1])
	batch_count = (total_size + batch_size - 1) // batch_size
	for batch_idx in range(batch_count):
		current_batch_size = batch_size																																													
		for (placeholder, tensor) in batch_tensors:
			batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]
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
mu_tf = tf.placeholder("float", shape=[]) # mu for LC
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
loss = cross_entropy + l2 * weight_decay
train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


################################################################################
######################## VARIABLES #############################################
################################################################################
variables = {}
weights = {}
for v in tf.trainable_variables():
	variables[v.name] = v
	if 'Batch' not in v.name:
		weights[v.name] = v

wC_tf = {}
for layer, _ in ref_weights.items():
	wC_tf[layer] = tf.placeholder("float",ref_weights[layer].get_shape())

lamda_tf = {}
for layer, _ in ref_weights.items():
	lamda_tf[layer] = tf.placeholder("float",ref_weights[layer].get_shape())

variables_init_placeholder = {}
for layer, _ in ref_variables.items():
	variables_init_placeholder[layer] = tf.placeholder("float", ref_variables[layer].get_shape())

w_init_placeholder = {}
for layer, _ in ref_weights.items():
	w_init_placeholder[layer] = tf.placeholder("float", ref_weights[layer].get_shape())

var_init = {}
for layer, _ in ref_variables.items():
	var_init[layer] = variables[layer].assign(variables_init_placeholder[layer])

w_init = {}
for layer, _ in ref_weights.items():
	w_init[layer] = weights[layer].assign(w_init_placeholder[layer])

norm_tf = tf.Variable(initial_value=[0.0], trainable=False)
for layer, _ in weights.items():
	norm_tf = norm_tf + tf.norm(weights[layer] - wC_tf[layer] - lamda_tf[layer] / mu_tf,ord='euclidean')

codebook_tf = {}
for layer, _ in ref_weights.items():
	codebook_tf[layer] = tf.Variable(tf.random_normal([k,1], stddev=0.01))

codebook_placeholder_tf = {}
for layer, _ in ref_weights.items():
	codebook_placeholder_tf[layer] = tf.placeholder("float", [k,1])

init_codebook_tf = {}
for layer, _ in ref_weights.items():
	init_codebook_tf[layer] = codebook_tf[layer].assign(codebook_placeholder_tf[layer])

Z_W_int_tf = {}
for layer, _ in ref_weights.items():
	Z_W_int_tf[layer] = tf.placeholder(tf.int32, [n_W[layer],k])

Z_W_tf = {}
for layer, _ in ref_weights.items():
	Z_W_tf[layer] = tf.cast(Z_W_int_tf[layer],tf.float32)

# DC retrain
W_DC_ret_tf = {}
for layer, _ in ref_weights.items():
	W_DC_ret_tf[layer] = tf.reshape(tf.matmul(Z_W_tf[layer] , codebook_tf[layer]), ref_weights[layer].get_shape())

# # Define loss and optimizer

# # Construct model using shared weights
# output_compression = model(x, wC_tf, biasC_tf)
# correct_prediction_compression = tf.equal(tf.argmax(output_compression, 1), tf.argmax(y, 1))
# accuracy_compression = tf.reduce_mean(tf.cast(correct_prediction_compression, tf.float32))
# loss_compression = tf.reduce_mean(
# 	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_compression))

# # DC retrain
# output_DC_retrain = model(x, W_DC_ret_tf,bias_DC_ret_tf)
# correct_prediction_DC_ret = tf.equal(tf.argmax(output_DC_retrain, 1), tf.argmax(y, 1))
# accuracy_DC_ret = tf.reduce_mean(tf.cast(correct_prediction_DC_ret, tf.float32))
# loss_DC_ret = tf.reduce_mean(
# 	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_DC_retrain))


regularizer = mu_tf / 2 * norm_tf
loss_L_step =  loss + regularizer 
train_L_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss_L_step)

###############################################################################
################### learn codebook and assignments ############################
###############################################################################
# flatten the weights and concatenate bias for each layer
w = {}
for layer, weight_matrix in ref_weights_values.items():
	if layer not in ['Variable_39:0']:
		w[layer] = weight_matrix.flatten().reshape(-1,1)
	elif layer = 'Variable_39:0':
		wf = ref_weights_values['Variable_39:0'].flatten()
		bf = ref_weights_values['Variable_40:0'].flatten()
		tmp = np.concatenate( (wf , bf) , axis=0)
		w[layer] = tmp.reshape(-1,1)


# dictionary to save the kmeans output for each layer 
kmeans = {}
# codebook of each layer i.e. centers of kmeans
C = {}
# assignments i.e. labels of kmeans
Z = {}
# quantized reference net i.e. prediction of kmeans
wC = {}

# Kmeans
for layer, _ in w.items():
	# wC[layer] = w[layer]
	# if ref_weights_values[layer].ndim != 1:
	kmeans[layer] = KMeans(n_clusters=k, random_state=0).fit(w[layer])
	C[layer] = kmeans[layer].cluster_centers_ 
	Z[layer] = kmeans[layer].labels_
	# quantize reference net
	if layer not in ['Variable_39:0','Variable_40:0']:
		wC[layer]= C[layer][Z[layer]]
	elif layer == 'Variable_39:0':
		wC[layer] = C[layer][Z[layer][:ref_weights_values[layer].size]]
	elif layer == 'Variable_40:0':
		wC[layer] = C['Variable_39:0'][Z['Variable_39:0'][ref_weights_values['Variable_39:0'].size:]]

C_DC = C
###############################################################################
########################## DC = Kmeans(w_bar) #################################
###############################################################################
################### TO SAVE DC MODEL ##################
model_file_name = 'DC_model_k_' + str(k) + '.ckpt'
model_file_path = './model/' + model_file_name 
####################### reshape weights #######################################
print('----------------------------------------------')
print('DC NET for k = {}' .format(k))
print('----------------------------------------------')

wC_reshape = {}
for layer, weight_matrix in wC.items():
	wC_reshape[layer] = wC[layer].reshape(ref_weights_values[layer].shape)

with tf.Session() as sess:
	batch_size = 64
	sess.run(tf.global_variables_initializer())
	# construct feed_dict
	feed_dict = {}
	for layer, _ in variables.items():
		feed_dict.update({ variables_init_placeholder[layer]: ref_values[layer] })
	sess.run(var_init,feed_dict=feed_dict)

	for layer, _ in weights.items():
		feed_dict.update({ w_init_placeholder[layer]: wC_reshape[layer] })
	sess.run(w_init,feed_dict=feed_dict)
	test_results = run_in_batch_avg(sess,[cross_entropy,accuracy],[x,y],
 				feed_dict = { 	x: data['test-data'], 
 								y: data['test-labels'], 
 								is_training: False, 
 								keep_prob: 1. })
	print('test results for DC: ', test_results)


	test_loss_DC = test_results[0]
	test_error_DC = 1-test_results[1]
###############################################################################
####################################### LC ####################################
# mu parameters
mu_0 = 0.001
a = 2.0
max_iter_each_L_step = 1000
LC_epoches = 21
batch_size = 64
minibatch = batch_size
batch_count = len(X_train) // batch_size
num_minibatches_data = batch_count

# ################### TO SAVE TRAINING AND TEST LOSS AND ERROR ##################
# ################### FOR REFERENCE NET #########################################
num_epoch_in_each_L_train = max_iter_each_L_step // batch_count
num_epoch_L_train = LC_epoches * ( num_epoch_in_each_L_train + 1)
epoch_L_train_vec = np.array(range(num_epoch_L_train))
epoch_LC_vec = np.array(range(LC_epoches)) 
train_loss_L = np.zeros(num_epoch_L_train + 1)
train_error_L = np.zeros(num_epoch_L_train + 1)
val_loss_L = np.zeros(num_epoch_L_train + 1)
val_error_L = np.zeros(num_epoch_L_train + 1)

test_loss_L = np.zeros(LC_epoches)
test_error_L = np.zeros(LC_epoches)

val_loss_C = np.zeros(LC_epoches)
val_error_C = np.zeros(LC_epoches)
test_loss_C = np.zeros(LC_epoches)
test_error_C = np.zeros(LC_epoches)

# ################### TO SAVE MODEL ##################
model_file_name = 'LC_model_k_' + str(k) + '.ckpt'
model_file_path = './model/' + model_file_name 

# initilize lambda == python reserved lambda so let's use lamda
lamda = {}
for layer, _ in ref_weights_values.items():
	lamda[layer] = np.zeros(ref_weights_values[layer].shape)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	batch_size = 64
	learning_rate = 0.001
	saver = tf.train.Saver()
	L_var_values = ref_values
	L_weights_values = ref_weights_values
	for j in range(LC_epoches):
		feed_dict = {}
		for layer, _ in variables.items():
			feed_dict.update({ variables_init_placeholder[layer]: L_var_values[layer] })
		
		for layer, _ in weights.items():
			feed_dict.update({ wC_tf[layer]: wC_reshape[layer] })
			feed_dict.update({ lamda_tf[layer]: lamda[layer] })
		sess.run(var_init,feed_dict=feed_dict)
		
		print('L step {} : ' .format(j))
		# adjust mu
		mu = mu_0 * ( a ** j )
		# adjust learning rate
		if k > 8:
			learning_rate = 0.001 * ( 0.98 ** j )
		else:
			learning_rate = 0.002 * ( 0.98 ** j )
		#######################################################################
		######## L Step #######################################################
		#######################################################################	
		print('----------------------------------------------')
		print('L STEP #{} for k = {}' .format(j,k))
		print('----------------------------------------------')

		# variable.initialized_value() ?
		for i in range(max_iter_each_L_step):
			X_train, y_train = data['train-data'], data['train-labels']
			index_minibatch = i % num_minibatches_data
			epoch = i // num_minibatches_data		
			# shuffle data at the begining of each epoch
			if index_minibatch == 0:
				X_train, y_train = shuffle_data(X_train, y_train)
			# mini batch 
			start_index = index_minibatch     * minibatch
			end_index   = (index_minibatch+1) * minibatch
			X_batch = X_train[start_index:end_index]
			y_batch = y_train[start_index:end_index]
			###################################################################
			####################### training batch in L #######################
			# construct feed_dict
			feed_dict = {}
			for layer, _ in weights.items():
				feed_dict.update({ wC_tf[layer]: wC_reshape[layer] })
				feed_dict.update({ lamda_tf[layer]: lamda[layer] })
				feed_dict.update({	x: X_batch,
						 		y: y_batch,
						 		lr: learning_rate,
						 		is_training: True, 
						 		keep_prob: 0.8,
						 		mu_tf: mu })

			batch_res = sess.run([ train_L_step,loss_L_step, accuracy ], 
								feed_dict = feed_dict)			
			if index_minibatch == 0:
				train_loss, train_accuracy = \
			 			sess.run([loss_L_step, accuracy], feed_dict = feed_dict)
				train_loss_L[epoch] = train_loss
				train_error_L[epoch] = 1 - train_accuracy
				print('L epoch: {}, train loss: {}, train error: {}' \
							.format(epoch, train_loss_L[epoch], train_error_L[epoch]) )
				feed_dict.update({ 	x: data['validation-data'], 
									y: data['validation-labels'], 
									is_training: False, 
									keep_prob: 1.,
									mu_tf: mu })
				validation_results = run_in_batch_avg(sess,[cross_entropy,accuracy],
												[x,y],feed_dict = feed_dict)
				val_loss_L[epoch] = validation_results[0]
				val_error_L[epoch] = 1 - validation_results[1]
				print('L epoch: {}, val loss: {}, val error: {}' \
							.format(epoch, val_loss_L[epoch], val_error_L[epoch]) )

		feed_dict.update({ 	x: data['test-data'], 
								y: data['test-labels'], 
								is_training: False, 
								keep_prob: 1.,
								mu_tf: mu })
		test_results = run_in_batch_avg(sess,[loss_L_step,accuracy],[x,y],
								feed_dict=feed_dict)
				
		test_loss_L[j] = test_results[0]
		test_error_L[j] = 1 - test_results[1]
		
		print('L step: {}, test loss: {}, test error: {}' \
							.format(j, test_loss_L[j], test_error_L[j]) )

		########################################################################
		########################## C STEP ######################################
		########################################################################
		########################################################################
		########### learn codebook and assignments #############################
		########################################################################
		# flatten the weights and concatenate bias for each layer
		L_var_values = {}
		L_weights_values = {}
		
		for layer, var in variables.items():
			L_var_values[layer] = sess.run(var)
			if 'Batch' not in layer:
				L_weights_values[layer] = sess.run(var)

		# flatten the weights and concatenate bias for each layer
		w = {}
		for layer, weight_matrix in L_weights_values.items():
			if layer not in ['Variable_39:0']:
				w[layer] = weight_matrix.flatten().reshape(-1,1)
			elif layer = 'Variable_39:0':
				wf = L_weights_values['Variable_39:0'].flatten()
				bf = L_weights_values['Variable_40:0'].flatten()
				tmp = np.concatenate( (wf , bf) , axis=0)
				w[layer] = tmp.reshape(-1,1)


		# dictionary to save the kmeans output for each layer 
		kmeans = {}
		# codebook of each layer i.e. centers of kmeans
		C = {}
		# assignments i.e. labels of kmeans
		Z = {}
		# quantized reference net i.e. prediction of kmeans
		wC = {}

		# Kmeans
		for layer, _ in w.items():
			# wC[layer] = w[layer]
			# if ref_weights_values[layer].ndim != 1:
			kmeans[layer] = KMeans(n_clusters=k, random_state=0).fit(w[layer])
			C[layer] = kmeans[layer].cluster_centers_ 
			Z[layer] = kmeans[layer].labels_
			# quantize reference net
			if layer not in ['Variable_39:0','Variable_40:0']:
				wC[layer]= C[layer][Z[layer]]
			elif layer == 'Variable_39:0':
				wC[layer] = C[layer][Z[layer][:ref_weights_values[layer].size]]
			elif layer == 'Variable_40:0':
				wC[layer] = C['Variable_39:0'][Z['Variable_39:0'][ref_weights_values['Variable_39:0'].size:]]

		wC_reshape = {}
		for layer, weight_matrix in wC.items():
			wC_reshape[layer] = wC[layer].reshape(ref_weights_values[layer].shape)

		######################################################################
		####################### accuracy using wc ############################
		feed_dict = {}
		for layer, _ in weights.items():
			feed_dict.update({ w_init_placeholder[layer]: wC_reshape[layer] })
		sess.run(w_init,feed_dict=feed_dict)
		test_results = run_in_batch_avg(sess,[cross_entropy,accuracy],[x,y],
 				feed_dict = { 	x: data['test-data'], 
 								y: data['test-labels'], 
 								is_training: False, 
 								keep_prob: 1. })
		print('test results for C: ', test_results)
		test_loss_C[j]= test_results[0]
		test_error_C[j] = 1 - test_results[1]
		print('C Step: {}, test loss: {}, test acuracy: {}' \
							.format(j, test_loss_C[j], test_error_C[j]) )
		#######################################################################
		############################ update lambda ############################
		for layer, _ in weights.items():
			lamda[layer] = lamda[layer] - mu * (L_weights_values[layer] - wC_reshape[layer])

		norm_compression = 0
		for layer, _ in w.items():
			if layer not in ['Variable_39:0']:
				norm_compression += LA.norm(w[layer] - wC[layer])
		elif layer = 'Variable_39:0':
				norm_compression += LA.norm(w[layer][:ref_weights_values[layer].size] - wC[layer])

		print('norm of compression: {} ' .format(norm_compression) )

		if norm_compression < 0.001:
			break

	save_path = saver.save(sess, model_file_path)
	C_LC = C

file_pickle = './results/results_pickle_k_' + str(k) + '.pkl'
with open(file_pickle,'wb') as f:
	pickle.dump(C_DC,f)
	pickle.dump(C_LC,f)
	pickle.dump(test_loss_DC,f)
	pickle.dump(test_error_DC,f)
	pickle.dump(train_loss_L,f)
	pickle.dump(train_error_L,f)
	pickle.dump(val_loss_L,f)
	pickle.dump(val_error_L,f)
	pickle.dump(test_loss_L,f)
	pickle.dump(test_error_L,f)
	pickle.dump(test_loss_C,f)
	pickle.dump(test_error_C,f)









