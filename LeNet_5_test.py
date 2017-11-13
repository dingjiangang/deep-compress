import tensorflow as tf

import input_MNIST_data
from input_MNIST_data import shuffle_data
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)

import numpy as np
import sys
import dill
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
print('architecture: LeNet-5 --- Data Set: MNIST')
print('----------------------------------------------')

print('----------------------------------------------')
print('Compression Algorithm for k = {}' .format(k))
print('----------------------------------------------')

# input and output shape
n_input   = data.train.images.shape[1]  # here MNIST data input (28,28)
n_classes = data.train.labels.shape[1]  # here MNIST (0-9 digits)

# dropout rate
dropout_rate = 0.5
# number of weights and bias in each layer
n_W = {}
n_b = {}

# network architecture hyper parameters
input_shape = [-1,28,28,1]
W0 = 28
H0 = 28

# Layer 1 -- conv
D1 = 1
F1 = 5
K1 = 20
S1 = 1
W1 = (W0 - F1) // S1 + 1
H1 = (H0 - F1) // S1 + 1
conv1_dim = [F1, F1, D1, K1]
conv1_strides = [1,S1,S1,1] 
n_W['conv1'] = F1 * F1 * D1 * K1
n_b['conv1'] = K1 

# Layer 2 -- max pool
D2 = K1
F2 = 2
K2 = D2
S2 = 2
W2 = (W1 - F2) // S2 + 1
H2 = (H1 - F2) // S2 + 1
layer2_ksize = [1,F2,F2,1]
layer2_strides = [1,S2,S2,1]

# Layer 3 -- conv
D3 = K2
F3 = 5
K3 = 50
S3 = 1
W3 = (W2 - F3) // S3 + 1
H3 = (H2 - F3) // S3 + 1
conv2_dim = [F3, F3, D3, K3]
conv2_strides = [1,S3,S3,1] 
n_W['conv2'] = F3 * F3 * D3 * K3
n_b['conv2'] = K3 

# Layer 4 -- max pool
D4 = K3
F4 = 2
K4 = D4
S4 = 2
W4 = (W3 - F4) // S4 + 1
H4 = (H3 - F4) // S4 + 1
layer4_ksize = [1,F4,F4,1]
layer4_strides = [1,S4,S4,1]


# Layer 5 -- fully connected
n_in_fc = W4 * H4 * D4
n_hidden = 500
fc_dim = [n_in_fc,n_hidden]
n_W['fc'] = n_in_fc * n_hidden
n_b['fc'] = n_hidden

# Layer 6 -- output
n_in_out = n_hidden
n_W['out'] = n_hidden * n_classes
n_b['out'] = n_classes

for key, value in n_W.items():
	n_W[key] = int(value)

for key, value in n_b.items():
	n_b[key] = int(value)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

learning_rate = tf.placeholder("float")
momentum_tf = tf.placeholder("float")
mu_tf = tf.placeholder("float")

# weights of LeNet-5 CNN -- tf tensors
weights = {
    # 5 x 5 convolution, 1 input image, 20 outputs
    'conv1': tf.Variable(tf.random_normal([F1, F1, D1, K1])),
    # 5x5 conv, 20 inputs, 50 outputs 
    'conv2': tf.Variable(tf.random_normal([F3, F3, D3, K3])),
    # fully connected, 800 inputs, 500 outputs
    'fc': tf.Variable(tf.random_normal([n_in_fc, n_hidden])),
    # 500 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

# biases of LeNet-5 CNN -- tf tensors
biases = {
    'conv1': tf.Variable(tf.random_normal([K1])),
    'conv2': tf.Variable(tf.random_normal([K3])),
    'fc': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def model(x,_W,_b):
	# Reshape input to a 4D tensor 
    x = tf.reshape(x, shape = input_shape)
    # LAYER 1 -- Convolution Layer
    conv1 = tf.nn.relu(tf.nn.conv2d(input = x, 
    								filter =_W['conv1'],
    								strides = [1,S1,S1,1],
    								padding = 'VALID') + _b['conv1'])
    # Layer 2 -- max pool
    conv1 = tf.nn.max_pool(	value = conv1, 
    						ksize = [1, F2, F2, 1], 
    						strides = [1, S2, S2, 1], 
    						padding = 'VALID')

    # LAYER 3 -- Convolution Layer
    conv2 = tf.nn.relu(tf.nn.conv2d(input = conv1, 
    								filter =_W['conv2'],
    								strides = [1,S3,S3,1],
    								padding = 'VALID') + _b['conv2'])
    # Layer 4 -- max pool
    conv2 = tf.nn.max_pool(	value = conv2 , 
    						ksize = [1, F4, F4, 1], 
    						strides = [1, S4, S4, 1], 
    						padding = 'VALID')
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.contrib.layers.flatten(conv2)
    fc = tf.nn.relu(tf.matmul(fc, _W['fc']) + _b['fc'])
    fc = tf.nn.dropout(fc, dropout_rate)

    output = tf.matmul(fc, _W['out']) + _b['out']
    output = tf.nn.dropout(output, keep_prob = dropout_rate)
    return output

wC_tf = {}
for layer, _ in weights.items():
	wC_tf[layer] = tf.placeholder("float", weights[layer].get_shape())

biasC_tf = {}
for layer, _ in biases.items():
	biasC_tf[layer] = tf.placeholder("float", biases[layer].get_shape())

lamda_tf = {}
for layer, _ in weights.items():
	lamda_tf[layer] = tf.placeholder("float", weights[layer].get_shape())

lamda_bias_tf = {}
for layer, _ in biases.items():
	lamda_bias_tf[layer] = tf.placeholder("float", biases[layer].get_shape())

w_init_placeholder = {}
for layer, _ in weights.items():
	w_init_placeholder[layer] = tf.placeholder("float", weights[layer].get_shape())

bias_init_placeholder = {}
for layer, _ in biases.items():
	bias_init_placeholder[layer] = tf.placeholder("float", biases[layer].get_shape())

w_init = {}
for layer, _ in weights.items():
	w_init[layer] = weights[layer].assign(w_init_placeholder[layer])

bias_init = {}
for layer, _ in biases.items():
	bias_init[layer] = biases[layer].assign(bias_init_placeholder[layer])

norm_tf = tf.Variable(initial_value=[0.0], trainable=False)
for layer, _ in weights.items():
	norm_tf = norm_tf + tf.norm(weights[layer] - wC_tf[layer] - lamda_tf[layer] / mu_tf,ord='euclidean')

for layer,_ in biases.items():
	norm_tf = norm_tf + tf.norm(biases[layer] - biasC_tf[layer] - lamda_bias_tf[layer] / mu_tf,ord='euclidean')

codebook_tf = {}
for layer, _ in weights.items():
	codebook_tf[layer] = tf.Variable(tf.random_normal([k,1], stddev=0.01))

codebook_placeholder_tf = {}
for layer, _ in weights.items():
	codebook_placeholder_tf[layer] = tf.placeholder("float", [k,1])

init_codebook_tf = {}
for layer, _ in weights.items():
	init_codebook_tf[layer] = codebook_tf[layer].assign(codebook_placeholder_tf[layer])

Z_W_int_tf = {}
for layer, _ in weights.items():
	Z_W_int_tf[layer] = tf.placeholder(tf.int32, [n_W[layer],k])

Z_W_tf = {}
for layer, _ in weights.items():
	Z_W_tf[layer] = tf.cast(Z_W_int_tf[layer],tf.float32)

Z_bias_int_tf = {}
for layer, _ in biases.items():
	Z_bias_int_tf[layer] = tf.placeholder(tf.int32, [n_b[layer],k])

Z_bias_tf = {}
for layer, _ in biases.items():
	Z_bias_tf[layer] = tf.cast(Z_bias_int_tf[layer],tf.float32)

# DC retrain
W_DC_ret_tf = {}
for layer, _ in weights.items():
	W_DC_ret_tf[layer] = tf.reshape(tf.matmul(Z_W_tf[layer] , codebook_tf[layer]), weights[layer].get_shape())

bias_DC_ret_tf = {}
for layer, _ in biases.items():
	bias_DC_ret_tf[layer] = tf.reshape(tf.matmul(Z_bias_tf[layer] , codebook_tf[layer]), biases[layer].get_shape())
# oe shape = -1

# Construct model
output = model(x,weights,biases)
# Softmax loss
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define loss and optimizer

# Construct model using shared weights
output_compression = model(x, wC_tf, biasC_tf)
correct_prediction_compression = tf.equal(tf.argmax(output_compression, 1), tf.argmax(y, 1))
accuracy_compression = tf.reduce_mean(tf.cast(correct_prediction_compression, tf.float32))
loss_compression = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_compression))

# DC retrain
output_DC_retrain = model(x, W_DC_ret_tf,bias_DC_ret_tf)
correct_prediction_DC_ret = tf.equal(tf.argmax(output_DC_retrain, 1), tf.argmax(y, 1))
accuracy_DC_ret = tf.reduce_mean(tf.cast(correct_prediction_DC_ret, tf.float32))
loss_DC_ret = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_DC_retrain))


regularizer = mu_tf / 2 * norm_tf

loss_L_step =  loss + regularizer 

# REFERENCE MODEL Parameters -- for training the Reference model: 

# Batch size
minibatch = 512
# Total minibatches
total_minibatches = 500
# number of minibatches in data
num_minibatches_data = data.train.images.shape[0] // minibatch

# Learning rate
lr = 0.01
# Learning rate decay:  every 2000 minibatches
learning_rate_decay = 0.98
learning_rate_stay_fixed = 2000

# Optimizer: Nesterov accelerated gradient with momentum 0.95
# this is for training the reference net
momentum = 0.9

optimizer = tf.train.MomentumOptimizer(
	learning_rate = learning_rate,
	momentum = momentum_tf,
	use_locking=False,
	name='Momentum',
	use_nesterov=True)

GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2
# GATE_OP:
# For each Op, make sure all gradients are computed before
# they are used.  This prevents race conditions for Ops that generate gradients
# for multiple inputs where the gradients depend on the inputs.
train = optimizer.minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name='train',
    grad_loss=None)


saver = tf.train.Saver()

train_L_step = optimizer.minimize(
    loss_L_step,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name='train_L_step',
    grad_loss=None)

train_DC_ret_step = optimizer.minimize(
    loss_DC_ret,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name='train_L_step',
    grad_loss=None)

init = tf.global_variables_initializer()

###############################################################################
######## training data and neural net architecture with weights w #############
###############################################################################
print('----------------------------------------------')
print('TRAINGING REFERENCE NET for k = {}' .format(k))
print('----------------------------------------------')
################### TO SAVE TRAINING AND TEST LOSS AND ERROR ##################
################### FOR REFERENCE NET #########################################
num_epoch_ref = total_minibatches // num_minibatches_data
epoch_ref_vec = np.array(range(num_epoch_ref+1)) 
train_loss_ref = np.zeros(num_epoch_ref+1)
train_error_ref = np.zeros(num_epoch_ref+1)
val_loss_ref = np.zeros(num_epoch_ref+1)
val_error_ref = np.zeros(num_epoch_ref+1)
test_loss_ref = np.zeros(num_epoch_ref+1)
test_error_ref = np.zeros(num_epoch_ref+1)

################### TO SAVE MODEL ##################
model_file_name = 'reference_model_k_' + str(k)
model_file_path = './model_lenet_5/' + model_file_name 

############################## TRAIN LOOP #####################################
with tf.Session() as sess:
	sess.run(init)
	for i in range(total_minibatches):
		index_minibatch = i % num_minibatches_data
		epoch = i // num_minibatches_data		
		# shuffle data at the begining of each epoch
		if index_minibatch == 0:
			X_train, y_train = shuffle_data(data)
		# adjust learning rate
		if i % learning_rate_stay_fixed == 0:
			j = i // learning_rate_stay_fixed
			if k > 8:
				lr = 0.01 * 0.98 ** j
			else:
				lr = 0.01 * 0.98 ** j
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]

		train.run(feed_dict = { x: X_batch,
					 			y: y_batch,
								learning_rate: lr,
								momentum_tf: momentum})
		
		############### LOSS AND ACCURACY EVALUATION ##########################
		if index_minibatch == 0:
			train_loss, train_accuracy = \
					sess.run([loss, accuracy], feed_dict = {x: X_batch, 
														    y: y_batch} )
			train_loss_ref[epoch] = train_loss
			train_error_ref[epoch] = 1 - train_accuracy

			val_loss, val_accuracy = \
			sess.run([loss, accuracy], feed_dict = {x: data.validation.images, 
													y: data.validation.labels} )
			val_loss_ref[epoch] = val_loss
			val_error_ref[epoch] = 1 - val_accuracy

			test_loss, test_accuracy = \
			sess.run([loss, accuracy], feed_dict = {x: data.test.images, 
													y: data.test.labels} )
			test_loss_ref[epoch] = test_loss
			test_error_ref[epoch] = 1 - test_accuracy

			print('step: {}, train loss: {}, train acuracy: {}' \
				.format(i, train_loss, train_accuracy) )
			print('step: {}, val loss: {}, val acuracy: {}' \
				.format(i, val_loss, val_accuracy) )
			print('step: {}, test loss: {}, test acuracy: {}' \
				.format(i, test_loss, test_accuracy) )
		#train_loss_ref = sess.run(loss)
		
	save_path = saver.save(sess, model_file_path)
	# reference weight and bias
	w_bar = sess.run(weights)
	bias_bar = sess.run(biases)

###############################################################################
################### learn codebook and assignments ############################
###############################################################################
# flatten the weights and concatenate bias for each layer
w = {}
for layer, weight_matrix in w_bar.items():
	wf = weight_matrix.flatten()
	wf = np.concatenate( (wf , bias_bar[layer]) , axis=0)
	w[layer] = wf.reshape(-1 , 1)

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
	kmeans[layer] = KMeans(n_clusters=k, random_state=0).fit(w[layer])
	C[layer] = kmeans[layer].cluster_centers_ 
	Z[layer] = kmeans[layer].labels_
	# quantize reference net
	wC[layer]= C[layer][Z[layer]]

C_DC = C

################### TO SAVE DC MODEL ##################
model_file_name = 'DC_model_k_' + str(k)
model_file_path = './model_lenet_5/' + model_file_name 

###############################################################################
########################## DC = Kmeans(w_bar) #################################
###############################################################################
####################### reshape weights #######################################
print('----------------------------------------------')
print('DC NET for k = {}' .format(k))
print('----------------------------------------------')

wC_reshape = {}
biasC = {}
for layer, _ in w_bar.items():
	wC_reshape[layer] = wC[layer][0:w_bar[layer].size].reshape(w_bar[layer].shape)
	biasC[layer] = wC[layer][w_bar[layer].size:].reshape(-1)

with tf.Session() as sess:
	sess.run(init)
	# construct feed_dict
	feed_dict = {}
	for layer, _ in weights.items():
		feed_dict.update({ wC_tf[layer]: wC_reshape[layer] })
		feed_dict.update({ biasC_tf[layer]: biasC[layer] })
	feed_dict.update({ 	x: data.validation.images,
				   		y: data.validation.labels})
	
	val_loss, val_accuracy = \
			sess.run([loss_compression, accuracy_compression], 
										feed_dict = feed_dict )
	val_loss_DC = val_loss
	val_error_DC = 1 - val_accuracy

	feed_dict.update({ 	x: data.test.images,
				   		y: data.test.labels})

	test_loss, test_accuracy = \
	sess.run([loss_compression, accuracy_compression], 
									feed_dict = feed_dict)
	test_loss_DC= test_loss
	test_error_DC = 1 - test_accuracy
	print('val loss: {}, val acuracy: {}' \
				.format(val_loss, val_accuracy) )
	print('test loss: {}, test acuracy: {}' \
				.format(test_loss, test_accuracy) )

	save_path = saver.save(sess, model_file_path)

###############################################################################
############################## DC WITH RETRAINING #############################
###############################################################################
print('----------------------------------------------')
print('DC WITH RETRAINING for k = {}' .format(k))
print('----------------------------------------------')

Z_W_matrix = {}

Z_bias_matrix = {}

# one hot matrix assignments for weights
for layer, _ in w_bar.items():
	tempZ = Z[layer][0:w_bar[layer].size]
	tempZ_mat = np.zeros([tempZ.size, k], dtype=np.int32)
	tempZ_mat[np.arange(tempZ.size), tempZ] = 1
	Z_W_matrix[layer] = tempZ_mat

# one hot matrix assignments for biases
for layer, _ in w_bar.items():
	tempZ = Z[layer][w_bar[layer].size:]
	tempZ_mat = np.zeros([tempZ.size, k], dtype=np.int32)
	tempZ_mat[np.arange(tempZ.size), tempZ] = 1
	Z_bias_matrix[layer] = tempZ_mat

total_minibatches = 500
num_epoch_DC_ret = total_minibatches // num_minibatches_data
epoch_DC_ret_vec = np.array(range(num_epoch_DC_ret+1)) 
train_loss_DC_ret = np.zeros(num_epoch_DC_ret+1)
train_error_DC_ret = np.zeros(num_epoch_DC_ret+1)
val_loss_DC_ret	 = np.zeros(num_epoch_DC_ret+1)
val_error_DC_ret = np.zeros(num_epoch_DC_ret+1)
test_loss_DC_ret = np.zeros(num_epoch_DC_ret+1)
test_error_DC_ret = np.zeros(num_epoch_DC_ret+1)

################### TO SAVE MODEL ##################
model_file_name = 'DC_ret_model_k_' + str(k)
model_file_path = './model_lenet_5/' + model_file_name 

with tf.Session() as sess:
	sess.run(init)
	
	feed_dict = {}
	for layer, _ in weights.items():
		feed_dict.update({ codebook_placeholder_tf[layer]: C[layer] })
	sess.run(init_codebook_tf, feed_dict= feed_dict)
	for i in range(total_minibatches):
		index_minibatch = i % num_minibatches_data
		epoch = i // num_minibatches_data		
		# shuffle data at the begining of each epoch
		if index_minibatch == 0:
			X_train, y_train = shuffle_data(data)
		# adjust learning rate
		if i % learning_rate_stay_fixed == 0:
			j = i // learning_rate_stay_fixed
			lr = 0.001 * learning_rate_decay ** j
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		
		# construct feed_dict
		feed_dict = {}
		for layer, _ in weights.items():
			feed_dict.update({Z_W_int_tf[layer]: Z_W_matrix[layer]})
			feed_dict.update({Z_bias_int_tf[layer]: Z_bias_matrix[layer]})
		feed_dict.update({	x: X_batch,
							y: y_batch,
							learning_rate: lr,
							momentum_tf: momentum})
	
		train_DC_ret_step.run(feed_dict = feed_dict)
		############### LOSS AND ACCURACY EVALUATION ##########################
		if index_minibatch == 0:
			train_loss, train_accuracy = \
					sess.run([loss_DC_ret, accuracy_DC_ret], feed_dict = feed_dict )
			train_loss_DC_ret[epoch] = train_loss
			train_error_DC_ret[epoch] = 1 - train_accuracy

			feed_dict.update( { x: data.validation.images, 
								y: data.validation.labels} )
			val_loss, val_accuracy = \
			sess.run([loss_DC_ret, accuracy_DC_ret], feed_dict = feed_dict )
			val_loss_DC_ret[epoch] = val_loss
			val_error_DC_ret[epoch] = 1 - val_accuracy

			feed_dict.update( { x: data.test.images, 
								y: data.test.labels} )

			test_loss, test_accuracy = \
			sess.run([loss_DC_ret, accuracy_DC_ret], feed_dict = feed_dict)
			test_loss_DC_ret[epoch] = test_loss
			test_error_DC_ret[epoch] = 1 - test_accuracy

			print('step: {}, train loss: {}, train acuracy: {}' \
							.format(i, train_loss, train_accuracy) )
			print('step: {}, val loss: {}, val acuracy: {}' \
							.format(i, val_loss, val_accuracy) )
			print('step: {}, test loss: {}, test acuracy: {}' \
							.format(i, test_loss, test_accuracy) )

		#train_loss_ref = sess.run(loss)
		
	save_path = saver.save(sess, model_file_path)
	# reference weight and bias
	C_DC_ret = sess.run(codebook_tf, feed_dict = feed_dict)


###############################################################################
###############################################################################
# initilize lambda == python reserved lambda so let's use lamda
lamda = {}
lamda_bias = {}
for layer, _ in w_bar.items():
	lamda[layer] = np.zeros(w_bar[layer].shape)
	lamda_bias[layer] = np.zeros(bias_bar[layer].shape).reshape(-1)

###############################################################################
####################################### LC ####################################
momentum = 0.95
# mu parameters
mu_0 = 9.75e-5
a = 1.1
max_iter_each_L_step = 300
LC_epoches = 2
random_w_init = 1 # 0: random init, 1 if init with reference net

################### TO SAVE TRAINING AND TEST LOSS AND ERROR ##################
################### FOR REFERENCE NET #########################################
num_epoch_in_each_L_train = max_iter_each_L_step // num_minibatches_data
num_epoch_L_train = LC_epoches * ( num_epoch_in_each_L_train + 1)
epoch_L_train_vec = np.array(range(num_epoch_L_train))
epoch_LC_vec = np.array(range(LC_epoches)) 
train_loss_L = np.zeros(num_epoch_L_train)
train_error_L = np.zeros(num_epoch_L_train)

val_loss_L = np.zeros(LC_epoches)
val_error_L = np.zeros(LC_epoches)
test_loss_L = np.zeros(LC_epoches)
test_error_L = np.zeros(LC_epoches)

val_loss_C = np.zeros(LC_epoches)
val_error_C = np.zeros(LC_epoches)
test_loss_C = np.zeros(LC_epoches)
test_error_C = np.zeros(LC_epoches)

################### TO SAVE MODEL ##################
model_file_name = 'LC_model_k_' + str(k)
model_file_path = './model_lenet_5/' + model_file_name 

with tf.Session() as sess: 
	###########################################################################
	######## Initilize weights and bias #######################################
	if random_w_init:
		# initilize weights and bias randomly
		sess.run(init)
	else:
		sess.run(init)
		# initilize weights and bias with reference net
		feed_dict = {}
		for layer, _ in weights.items():
			feed_dict.update({w_init_placeholder[layer]: w_bar[layer]})
			feed_dict.update({bias_init_placeholder[layer]: bias_bar[layer]})

		sess.run([w_init,bias_init], feed_dict=feed_dict)
	
	for j in range(LC_epoches):
		print('L step {} : ' .format(j))
		# adjust mu
		mu = mu_0 * ( a ** j )
		# adjust learning rate
		if k > 8:
			lr = 0.01 * ( 0.98 ** j )
		else:
			lr = 0.01 * ( 0.98 ** j )
		#######################################################################
		######## L Step #######################################################
		#######################################################################	
		print('----------------------------------------------')
		print('L STEP #{} for k = {}' .format(j,k))
		print('----------------------------------------------')

		# variable.initialized_value() ?
		for i in range(max_iter_each_L_step):
			index_minibatch = i % num_minibatches_data
			epoch = i // num_minibatches_data		
			# shuffle data at the begining of each epoch
			if index_minibatch == 0:
				X_train, y_train = shuffle_data(data)
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
				feed_dict.update({ biasC_tf[layer]: biasC[layer] })
				feed_dict.update({ lamda_tf[layer]: lamda[layer] })
				feed_dict.update({ lamda_bias_tf[layer]: lamda_bias[layer] })
			feed_dict.update({	x: X_batch,
						 		y: y_batch,
						 		learning_rate: lr,
						 		momentum_tf: momentum,
						 		mu_tf: mu }),
			# train on batch
			train_L_step.run(feed_dict)
			
			if index_minibatch == 0:
				train_loss, train_accuracy = \
				 		sess.run([loss_L_step, accuracy], feed_dict = feed_dict)
				index = j*num_epoch_in_each_L_train+epoch
				train_loss_L[index] = train_loss
				train_error_L[index] = 1 - train_accuracy

				feed_dict.update( { x: data.validation.images, 
									y: data.validation.labels })
				val_loss, val_accuracy = \
					sess.run([loss_L_step, accuracy], feed_dict = feed_dict)

				feed_dict.update( { x: data.test.images, 
									y: data.test.labels })

				test_loss, test_accuracy = \
					sess.run([loss_L_step, accuracy], feed_dict = feed_dict)

				print('step: {}, train loss: {}, train acuracy: {}' \
							.format(i, train_loss, train_accuracy) )
				print('step: {}, val loss: {}, val acuracy: {}' \
							.format(i, val_loss, val_accuracy) )
				print('step: {}, test loss: {}, test acuracy: {}' \
							.format(i, test_loss, test_accuracy) )
			# reference weight and bias
			w_bar = sess.run(weights)
			bias_bar = sess.run(biases)
		######################################################################
		####################### accuracy using w #############################
		feed_dict = {}
		for layer, _ in weights.items():
			feed_dict.update({wC_tf[layer]: wC_reshape[layer]})
			feed_dict.update({biasC_tf[layer]: biasC[layer]})
			feed_dict.update({lamda_tf[layer]: lamda[layer]})
			feed_dict.update({lamda_bias_tf[layer]: lamda_bias[layer]})
		feed_dict.update({	mu_tf: mu,
							x: data.validation.images, 
							y: data.validation.labels})

		val_loss, val_accuracy = \
		sess.run([loss_L_step, accuracy], feed_dict = feed_dict)
		val_loss_L[j] = val_loss
		val_error_L[j] = 1 - val_accuracy

		feed_dict.update({	x: data.test.images, 
							y: data.test.labels})
	
		test_loss, test_accuracy = \
		sess.run([loss_L_step, accuracy], feed_dict = feed_dict)
		test_loss_L[j] = test_loss
		test_error_L[j] = 1 - test_accuracy
		print('L step: {}, val loss: {}, val acuracy: {}' \
							.format(j, val_loss, val_accuracy) )
		print('L step: {}, test loss: {}, test acuracy: {}' \
							.format(j, test_loss, test_accuracy) )
		#######################################################################
		######## C Step #######################################################
		#######################################################################
		print('----------------------------------------------')
		print('C STEP #{} for k = {}' .format(j,k))
		print('----------------------------------------------')
		# flatten the weights and concatenate bias for each layer
		w = {}
		for layer, _ in w_bar.items():
			wf = w_bar[layer].flatten() - lamda[layer].flatten() / mu
			bf = bias_bar[layer] - lamda_bias[layer] / mu
			wf = np.concatenate( (wf , bf) , axis=0)
			w[layer] = wf.reshape(-1 , 1)

		# Kmeans
		for layer, _ in w.items():
			kmeans[layer] = KMeans(n_clusters=k, random_state=0).fit(w[layer])
			C[layer] = kmeans[layer].cluster_centers_ 
			Z[layer] = kmeans[layer].labels_
			# quantize reference net
			wC[layer]= C[layer][Z[layer]]
		######################################################################
		####################### reshape weights ##############################
		for layer, _ in w_bar.items():
			wC_reshape[layer] = wC[layer][0:w_bar[layer].size].reshape(w_bar[layer].shape)
			biasC[layer] = wC[layer][w_bar[layer].size:].reshape(-1)
		
		######################################################################
		####################### accuracy using wc ############################
		feed_dict = {}
		for layer, _ in weights.items():
			feed_dict.update({wC_tf[layer]: wC_reshape[layer]})
			feed_dict.update({biasC_tf[layer]: biasC[layer]})
		feed_dict.update({	x: data.validation.images, 
							y: data.validation.labels })

		val_loss, val_accuracy = \
			sess.run([loss_compression, accuracy_compression], 
							feed_dict = feed_dict)

		feed_dict.update({	x: data.test.images, 
							y: data.test.labels })

		val_loss_C[j] = val_loss
		val_error_C[j] = 1 - val_accuracy

		test_loss, test_accuracy = \
		sess.run([loss_compression, accuracy_compression], feed_dict = feed_dict)
		test_loss_C[j]= test_loss
		test_error_C[j] = 1 - test_accuracy
		print('val loss: {}, val acuracy: {}' \
							.format(val_loss, val_accuracy) )
		print('test loss: {}, test acuracy: {}' \
							.format(test_loss, test_accuracy) )
		#######################################################################
		############################ update lambda ############################
		for layer, _ in w_bar.items():
			lamda[layer] = lamda[layer] - mu * (w_bar[layer] - wC_reshape[layer])
			lamda_bias[layer] = lamda_bias[layer] - mu * (bias_bar[layer] - biasC[layer])

		norm_compression = 0
		for layer, _ in w_bar.items():
			norm_compression = LA.norm(w[layer] - wC[layer])

		print('norm of compression: {} ' .format(norm_compression) )

		# if norm_compression < 0.001:
		# 	break

	save_path = saver.save(sess, model_file_path)
	C_LC = C

###############################################################################
############################## LC WITH RETRAINING #############################
###############################################################################
print('----------------------------------------------')
print('LC with RETRAINING for k = {}' .format(k))
print('----------------------------------------------')

Z_W_matrix = {}

Z_bias_matrix = {}

# one hot matrix assignments for weights
for layer, _ in w_bar.items():
	tempZ = Z[layer][0:w_bar[layer].size]
	tempZ_mat = np.zeros([tempZ.size, k], dtype=np.int32)
	tempZ_mat[np.arange(tempZ.size), tempZ] = 1
	Z_W_matrix[layer] = tempZ_mat

# one hot matrix assignments for biases
for layer, _ in w_bar.items():
	tempZ = Z[layer][w_bar[layer].size:]
	tempZ_mat = np.zeros([tempZ.size, k], dtype=np.int32)
	tempZ_mat[np.arange(tempZ.size), tempZ] = 1
	Z_bias_matrix[layer] = tempZ_mat

total_minibatches = 500
num_epoch_LC_ret = total_minibatches // num_minibatches_data
epoch_LC_ret_vec = np.array(range(num_epoch_LC_ret+1)) 
train_loss_LC_ret = np.zeros(num_epoch_LC_ret+1)
train_error_LC_ret = np.zeros(num_epoch_LC_ret+1)
val_loss_LC_ret	 = np.zeros(num_epoch_LC_ret+1)
val_error_LC_ret = np.zeros(num_epoch_LC_ret+1)
test_loss_LC_ret = np.zeros(num_epoch_LC_ret+1)
test_error_LC_ret = np.zeros(num_epoch_LC_ret+1)

################### TO SAVE MODEL ##################
model_file_name = 'LC_ret_model_k_' + str(k)
model_file_path = './model_lenet_5/' + model_file_name 

with tf.Session() as sess:
	sess.run(init)
	
	feed_dict = {}
	for layer, _ in weights.items():
		feed_dict.update({codebook_placeholder_tf[layer]: C[layer]})
	
	sess.run(init_codebook_tf, feed_dict= feed_dict)
	for i in range(total_minibatches):
		index_minibatch = i % num_minibatches_data
		epoch = i // num_minibatches_data		
		# shuffle data at the begining of each epoch
		if index_minibatch == 0:
			X_train, y_train = shuffle_data(data)
		# adjust learning rate
		if i % learning_rate_stay_fixed == 0:
			j = i // learning_rate_stay_fixed
			lr = 0.001 * learning_rate_decay ** j
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		
		feed_dict = {}
		for layer, _ in weights.items():
			feed_dict.update({Z_W_int_tf[layer]: Z_W_matrix[layer]})
			feed_dict.update({Z_bias_int_tf[layer]: Z_bias_matrix[layer]})
		feed_dict.update({	x: X_batch,
							y: y_batch,
							learning_rate: lr,
							momentum_tf: momentum})
		
		train_DC_ret_step.run(feed_dict = feed_dict)
		############### LOSS AND ACCURACY EVALUATION ##########################
		if index_minibatch == 0:
			train_loss, train_accuracy = \
					sess.run([loss_DC_ret, accuracy_DC_ret], feed_dict = feed_dict )
			train_loss_LC_ret[epoch] = train_loss
			train_error_LC_ret[epoch] = 1 - train_accuracy

			feed_dict.update( { x: data.validation.images, 
								y: data.validation.labels} )
			val_loss, val_accuracy = \
			sess.run([loss_DC_ret, accuracy_DC_ret], feed_dict = feed_dict )
			val_loss_LC_ret[epoch] = val_loss
			val_error_LC_ret[epoch] = 1 - val_accuracy

			feed_dict.update( { x: data.test.images, 
								y: data.test.labels} )

			test_loss, test_accuracy = \
			sess.run([loss_DC_ret, accuracy_DC_ret], feed_dict = feed_dict)
			test_loss_LC_ret[epoch] = test_loss
			test_error_LC_ret[epoch] = 1 - test_accuracy
			
			print('step: {}, train loss: {}, train acuracy: {}' \
							.format(i, train_loss, train_accuracy) )
			print('step: {}, val loss: {}, val acuracy: {}' \
							.format(i, val_loss, val_accuracy) )
			print('step: {}, test loss: {}, test acuracy: {}' \
							.format(i, test_loss, test_accuracy) )

		#train_loss_ref = sess.run(loss)
		
	save_path = saver.save(sess, model_file_path)
	# reference weight and bias
	C_LC_ret = sess.run(codebook_tf, feed_dict = feed_dict)


df_ref = pd.DataFrame({	'train_error_ref' : train_loss_ref,
						'train_error_ref': train_error_ref,
						'val_loss_ref': val_loss_ref,
						'val_error_ref': val_error_ref,
						'test_loss_ref': test_loss_ref,
						'test_error_ref': test_error_ref})

df_DC = pd.DataFrame({	'val_loss_DC': val_loss_DC,
						'val_error_DC': val_error_DC,
						'test_loss_DC': test_loss_DC,
						'test_error_DC': test_error_DC}, index=[0])


df_DC_ret = pd.DataFrame({	'val_loss_DC_ret': val_loss_DC_ret,
							'val_error_DC_ret': val_error_DC_ret,
							'test_loss_DC_ret': test_loss_DC_ret,
							'test_error_DC_ret': test_error_DC_ret})

df_L_train = pd.DataFrame({	'train_loss_L' : train_loss_L,
							'train_error_L': train_error_L})

df_LC = pd.DataFrame({	'val_loss_L': val_loss_L,
						'val_error_L': val_error_L,
						'test_loss_L': test_loss_L,
						'test_error_L': test_error_L,
						'val_loss_C': val_loss_C,
						'val_error_C': val_error_C,
						'test_loss_C': test_loss_C,
						'test_error_C': test_error_C})

df_LC_ret = pd.DataFrame({	'train_loss_LC_ret': train_loss_LC_ret,
							'train_error_LC_ret': train_error_LC_ret,
							'val_loss_LC_ret': val_loss_LC_ret,
							'val_error_LC_ret': val_error_LC_ret,
							'test_loss_LC_ret': test_loss_LC_ret,
							'test_error_LC_ret': test_error_LC_ret})

file_pickle = './results_lenet_5/results_pickle_k_' + str(k) + '.pkl'
with open(file_pickle,'wb') as f:
	pickle.dump(C_DC,f)
	pickle.dump(C_DC_ret,f)
	pickle.dump(C_LC,f)
	pickle.dump(C_LC_ret,f)
	df_ref.to_pickle(f)
	df_DC.to_pickle(f)
	df_DC_ret.to_pickle(f)
	df_L_train.to_pickle(f)
	df_LC.to_pickle(f)
	df_LC_ret.to_pickle(f)

import pickle
file_pickle = './results_lenet_5/results_pickle_k_' + str(k) + '.pkl'
with open(file_pickle,'rb') as f:
	C_DC = pickle.load(f)
	C_DC_ret = pickle.load(f)
	C_LC = pickle.load(f)
	C_LC_ret = pickle.load(f)
	df_ref = pickle.load(f)
	df_DC = pickle.load(f)
	df_DC_ret = pickle.load(f)
	df_L_train = pickle.load(f)
	df_LC = pickle.load(f)
	df_LC_ret = pickle.load(f)