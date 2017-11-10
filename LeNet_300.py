import tensorflow as tf

import input_MNIST_data
from input_MNIST_data import shuffle_data
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)

import numpy as np
import sys

import pickle

import pandas as pd

# df_ref = pd.DataFrame({'iter' : [],
# 					   'loss_train_ref': [],
# 					   'error_train_ref': [],
# 					   'loss_test_ref': [],
# 					   'error_test_ref':[]})

# df_DC = pd.DataFrame({ 'loss_test_DC': [],
# 					   'error_test_DC': [] })

# df_DC_retrain = pd.DataFrame({ 'iter' : [],
# 							   'loss_train_DC_retrain': [],
# 							   'error_train_DC_retrain': [],
# 							   'loss_test_DC_retrain': [],
# 							   'error_test_DC_retrain':[]} )

# df_LC = pd.DataFrame({ 'iter' : [],
# 					   'loss_train_L': [],
# 					   'error_train_L': [],
# 					   'loss_test_L': [],
# 					   'error_test_L':[],
# 					   'loss_test_C': [],
# 					   'error_test_C': [] })


import argparse
# Set up argument parser
ap = argparse.ArgumentParser()
# Single positional argument, nargs makes it optional
ap.add_argument('k', nargs='?', default=2)
# codebook size
k = ap.parse_args().k


from sklearn.cluster import KMeans
from numpy import linalg as LA

# input and output shape
n_input   = data.train.images.shape[1]  # here MNIST data input (28,28)
n_classes = data.train.labels.shape[1]  # here MNIST (0-9 digits)

# Network Parameters
n_hidden_1 = 300  # 1st layer num features
n_hidden_2 = 100  # 2nd layer num features


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
learning_rate = tf.placeholder("float")
momentum_tf = tf.placeholder("float")


def model(_X, _W, _bias):
    # Hidden layer with tanh activation
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _W['fc1']), _bias['fc1']))  
    # Hidden layer with tanh activation
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, _W['fc2']), _bias['fc2']))  
    # output without any activation
    output = tf.add(tf.matmul(layer_2, _W['out']) , _bias['out'])
    return output

def model_compression(_X, _wC_tf, _biasC_tf):
    # Hidden layer with tanh activation
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _wC_tf['fc1']), _biasC_tf['fc1']))  
    # Hidden layer with tanh activation
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, _wC_tf['fc2']), _biasC_tf['fc2']))  
    # output without any activation
    output_compression = tf.add(tf.matmul(layer_2, _wC_tf['out']) , _biasC_tf['out'])
    return output_compression

def model_compression_retrain(_X, _W_DC_ret_tf, _bias_DC_ret_tf):
    # Hidden layer with tanh activation
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _W_DC_ret_tf['fc1']), _bias_DC_ret_tf['fc1']))  
    # Hidden layer with tanh activation
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, _W_DC_ret_tf['fc2']), _bias_DC_ret_tf['fc2']))  
    # output without any activation
    output_DC_retrain = tf.add(tf.matmul(layer_2, _W_DC_ret_tf['out']) , _bias_DC_ret_tf['out'])
    return output_DC_retrain


# Store layers weight & bias
W = {
    'fc1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}

bias = {
    'fc1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}

# L step tf
mu_tf = tf.placeholder("float")
wC_tf = {
    'fc1': tf.placeholder("float", [n_input, n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_1, n_hidden_2]),
    'out': tf.placeholder("float", [n_hidden_2, n_classes])
}
biasC_tf = {
	'fc1': tf.placeholder("float", [n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_2]),
    'out': tf.placeholder("float", [n_classes])
}

lamda_tf = {
	'fc1': tf.placeholder("float", [n_input, n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_1, n_hidden_2]),
    'out': tf.placeholder("float", [n_hidden_2, n_classes])
}

lamda_bias_tf = {
	'fc1': tf.placeholder("float", [n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_2]),
    'out': tf.placeholder("float", [n_classes])
}

w_init_placeholder = {
    'fc1': tf.placeholder("float", [n_input, n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_1, n_hidden_2]),
    'out': tf.placeholder("float", [n_hidden_2, n_classes])
}
bias_init_placeholder = {
	'fc1': tf.placeholder("float", [n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_2]),
    'out': tf.placeholder("float", [n_classes])
}

w_init = {
 	'fc1' : W['fc1'].assign(w_init_placeholder['fc1']),
 	'fc2' : W['fc2'].assign(w_init_placeholder['fc2']),
 	'out' : W['out'].assign(w_init_placeholder['out'])
}

bias_init = {
 	'fc1' : bias['fc1'].assign(bias_init_placeholder['fc1']),
 	'fc2' : bias['fc2'].assign(bias_init_placeholder['fc2']),
 	'out' : bias['out'].assign(bias_init_placeholder['out'])
}

norm_tf = tf.norm( W['fc1'] - wC_tf['fc1'] - lamda_tf['fc1'] / mu_tf ,ord='euclidean') \
	    + tf.norm( W['fc2'] - wC_tf['fc2'] - lamda_tf['fc2'] / mu_tf ,ord='euclidean') \
	    + tf.norm( W['out'] - wC_tf['out'] - lamda_tf['out'] / mu_tf,ord='euclidean') \
	    + tf.norm( bias['fc1'] - biasC_tf['fc1'] - lamda_bias_tf['fc1'] / mu_tf ,ord='euclidean') \
	    + tf.norm( bias['fc2'] - biasC_tf['fc2'] - lamda_bias_tf['fc2'] / mu_tf ,ord='euclidean') \
	    + tf.norm( bias['out'] - biasC_tf['out'] - lamda_bias_tf['out'] / mu_tf ,ord='euclidean')


codebook_tf = {
	'fc1' : tf.Variable(tf.random_normal([k,1], stddev=0.01)),
	'fc2' : tf.Variable(tf.random_normal([k,1], stddev=0.01)),
	'out' : tf.Variable(tf.random_normal([k,1], stddev=0.01))
}

codebook_placeholder_tf = {
	'fc1': tf.placeholder("float", [k,1]),
	'fc2': tf.placeholder("float", [k,1]),
	'out': tf.placeholder("float", [k,1]),
}

init_codebook_tf = {
	'fc1': codebook_tf['fc1'].assign(codebook_placeholder_tf['fc1']),
	'fc2': codebook_tf['fc1'].assign(codebook_placeholder_tf['fc2']),
	'out': codebook_tf['out'].assign(codebook_placeholder_tf['out']),
}

#
Z_W_int_tf = {
	'fc1': tf.placeholder(tf.int32, [n_input * n_hidden_1, k]),
    'fc2': tf.placeholder(tf.int32, [n_hidden_1 * n_hidden_2, k]),
    'out': tf.placeholder(tf.int32, [n_hidden_2 * n_classes, k])
}

Z_W_tf = {
	'fc1': tf.cast(Z_W_int_tf['fc1'],tf.float32),
    'fc2': tf.cast(Z_W_int_tf['fc2'],tf.float32),
    'out': tf.cast(Z_W_int_tf['out'],tf.float32)
}

Z_bias_int_tf = {
	'fc1': tf.placeholder(tf.int32, [n_hidden_1,k]),
    'fc2': tf.placeholder(tf.int32, [n_hidden_2,k]),
    'out': tf.placeholder(tf.int32, [n_classes,k])
}

Z_bias_tf = {
	'fc1': tf.cast(Z_bias_int_tf['fc1'],tf.float32),
    'fc2': tf.cast(Z_bias_int_tf['fc2'],tf.float32),
    'out': tf.cast(Z_bias_int_tf['out'],tf.float32)
}
# DC retrain
W_DC_ret_tf = {
	'fc1': tf.reshape(tf.matmul(Z_W_tf['fc1'] , codebook_tf['fc1']), [n_input, n_hidden_1]),
	'fc2': tf.reshape(tf.matmul(Z_W_tf['fc2'] , codebook_tf['fc2']), [n_hidden_1, n_hidden_2]),
	'out': tf.reshape(tf.matmul(Z_W_tf['out'] , codebook_tf['out']), [n_hidden_2, n_classes])
}

bias_DC_ret_tf = {
	'fc1': tf.reshape( tf.matmul(Z_bias_tf['fc1'], codebook_tf['fc1']), [-1]),
	'fc2': tf.reshape( tf.matmul(Z_bias_tf['fc2'], codebook_tf['fc2']), [-1]),
	'out': tf.reshape( tf.matmul(Z_bias_tf['out'], codebook_tf['out']), [-1])
}


# Construct model
output = model(x, W, bias)
# Define loss and optimizer
# Softmax loss
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Construct model using shared weights
output_compression = model_compression(x, wC_tf, biasC_tf)
correct_prediction_compression = tf.equal(tf.argmax(output_compression, 1), tf.argmax(y, 1))
accuracy_compression = tf.reduce_mean(tf.cast(correct_prediction_compression, tf.float32))
loss_compression = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_compression))

# DC retrain
output_DC_retrain = model_compression_retrain(x, W_DC_ret_tf,bias_DC_ret_tf)
correct_prediction_DC_ret = tf.equal(tf.argmax(output_DC_retrain, 1), tf.argmax(y, 1))
accuracy_DC_ret = tf.reduce_mean(tf.cast(correct_prediction_DC_ret, tf.float32))
loss_DC_ret = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_DC_retrain))


regularizer = mu_tf / 2 * norm_tf

loss_L_step =  loss + regularizer 

# learning_rate = tf.placeholder("float")

# grad_w = {
#     'fc1': tf.gradients(loss, W['fc1']),
#     'fc2': tf.gradients(loss, W['fc2']),
#     'out': tf.gradients(loss, W['out'])
# }

# grad_bias = {
#     'fc1': tf.gradients(loss, bias['fc1']),
#     'fc2': tf.gradients(loss, bias['fc1']),
#     'out': tf.gradients(loss, bias['fc1'])
# }

# new_W = {
# 	'fc1' : W['fc1'].assign(W['fc1'] - learning_rate * grad_w['fc1']),
# 	'fc2' : W['fc2'].assign(W['fc2'] - learning_rate * grad_w['fc2']),
# 	'out' : W['out'].assign(W['out'] - learning_rate * grad_w['out'])
# }

# new_bias = {
# 	'fc1' : bias['fc1'].assign(bias['fc1'] - learning_rate * grad_bias['fc1']),
# 	'fc2' : bias['fc2'].assign(bias['fc2'] - learning_rate * grad_bias['fc2']),
# 	'out' : W['out'].assign(bias['out'] - learning_rate * grad_bias['out'])
# }


#Training the Reference model: 

# Batch size: 512
minibatch = 512
# Total minibatches
total_minibatches = 200
# number of minibatches in data
num_minibatches_data = data.train.images.shape[0] // minibatch

# Learning rate
lr = 0.02
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
model_file_path = './model/' + model_file_name 

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
			lr = learning_rate_decay ** j
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		
		# if i % 100 == 0:
		# 	train_accuracy = accuracy.eval(
		# 		feed_dict={x: X_batch, 
		# 				   y: y_batch})
		# 	print('step {}, training accuracy {}' .format(i, train_accuracy))

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

		
		train.run(feed_dict = { x: X_batch,
					 			y: y_batch,
								learning_rate: lr,
								momentum_tf: momentum})
		#train_loss_ref = sess.run(loss)
		
	save_path = saver.save(sess, model_file_path)
	# reference weight and bias
	w_bar = sess.run(W)
	bias_bar = sess.run(bias)

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


################### TO SAVE DC MODEL ##################
model_file_name = 'DC_model_k_' + str(k)
model_file_path = './model/' + model_file_name 

###############################################################################
########################## DC = Kmeans(w_bar) #################################
###############################################################################
####################### reshape weights #######################################
wC_reshape = {}
biasC = {}
for layer, _ in w_bar.items():
	wC_reshape[layer] = wC[layer][0:w_bar[layer].size].reshape(w_bar[layer].shape)
	biasC[layer] = wC[layer][w_bar[layer].size:].reshape(-1)
	# C[layer] = C[layer]

with tf.Session() as sess:
	sess.run(init)
	feed_dict = {
			wC_tf['fc1']: wC_reshape['fc1'],
			wC_tf['fc2']: wC_reshape['fc2'],
			wC_tf['out']: wC_reshape['out'],
			biasC_tf['fc1']: biasC['fc1'],
			biasC_tf['fc2']: biasC['fc2'],
			biasC_tf['out']: biasC['out'],
			x: data.validation.images,
			y: data.validation.labels}
	
	val_loss, val_accuracy = \
			sess.run([loss_compression, accuracy_compression], 
							feed_dict = feed_dict )
	val_loss_DC = val_loss
	val_error_DC = 1 - val_accuracy

	feed_dict = {
			wC_tf['fc1']: wC_reshape['fc1'],
			wC_tf['fc2']: wC_reshape['fc2'],
			wC_tf['out']: wC_reshape['out'],
			biasC_tf['fc1']: biasC['fc1'],
			biasC_tf['fc2']: biasC['fc2'],
			biasC_tf['out']: biasC['out'],
			x: data.test.images, 
			y: data.test.labels}


	test_loss, test_accuracy = \
	sess.run([loss_compression, accuracy_compression], 
								feed_dict = feed_dict)
	test_loss_DC= test_loss
	test_error_DC = 1 - test_accuracy
	save_path = saver.save(sess, model_file_path)

###############################################################################
############################## DC WITH RETRAINING #############################
###############################################################################

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
model_file_path = './model/' + model_file_name 

with tf.Session() as sess:
	sess.run(init)
	feed_dict = {
		codebook_placeholder_tf['fc1']: C['fc1'],
		codebook_placeholder_tf['fc2']: C['fc2'],
		codebook_placeholder_tf['out']: C['out']
	}
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
			lr = learning_rate_decay ** j
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		
		feed_dict = {
			Z_W_int_tf['fc1']: Z_W_matrix['fc1'],
			Z_W_int_tf['fc2']: Z_W_matrix['fc2'],
			Z_W_int_tf['out']: Z_W_matrix['out'],
			Z_bias_int_tf['fc1']: Z_bias_matrix['fc1'],
			Z_bias_int_tf['fc2']: Z_bias_matrix['fc2'],
			Z_bias_int_tf['out']: Z_bias_matrix['out'],
			x: X_batch,
			y: y_batch,
			learning_rate: lr,
			momentum_tf: momentum
		}
		train_DC_ret_step.run(feed_dict = feed_dict)
		# if i % 100 == 0:
		# 	train_accuracy = accuracy.eval(
		# 		feed_dict={x: X_batch, 
		# 				   y: y_batch})
		# 	print('step {}, training accuracy {}' .format(i, train_accuracy))

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
max_iter_each_L_step = 2000
LC_epoches = 2
random_w_init = 0 # 0: random init, 1 if init with reference net

################### TO SAVE TRAINING AND TEST LOSS AND ERROR ##################
################### FOR REFERENCE NET #########################################
num_epoch_in_each_L_train = max_iter_each_L_step // 100
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
model_file_path = './model/' + model_file_name 

with tf.Session() as sess: 
	###########################################################################
	######## Initilize weights and bias #######################################
	if random_w_init:
		# initilize weights and bias randomly
		sess.run(init)
	else:
		sess.run(init)
		# initilize weights and bias with reference net
		feed_dict = {
			w_init_placeholder['fc1']: w_bar['fc1'],
			w_init_placeholder['fc2']: w_bar['fc2'],
			w_init_placeholder['out']: w_bar['out'],
			bias_init_placeholder['fc1']: bias_bar['fc1'],
			bias_init_placeholder['fc2']: bias_bar['fc2'],
			bias_init_placeholder['out']: bias_bar['out']
		}
		sess.run([w_init,bias_init], feed_dict=feed_dict)
	
	for j in range(LC_epoches):
		print('L step {} : ' .format(j))
		# adjust mu
		mu = mu_0 * ( a ** j )
		# adjust learning rate
		lr = 0.1 * ( 0.99 ** j )
		#######################################################################
		######## L Step #######################################################
		#######################################################################	
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
		
			epoch_train = i // 100
			if i % 100 == 0:
				train_accuracy = accuracy.eval(
					feed_dict = {x: X_batch, 
						   		 y: y_batch})
				print('step {}, training accuracy {}' .format(i, train_accuracy))

			###################################################################
			####################### training batch in L #######################
			# train on batch
			feed_dict = {x: X_batch,
						 y: y_batch,
						 learning_rate: lr,
						 momentum_tf: momentum,
						 mu_tf: mu,
						 wC_tf['fc1']: wC_reshape['fc1'],
						 wC_tf['fc2']: wC_reshape['fc2'],
						 wC_tf['out']: wC_reshape['out'],
						 biasC_tf['fc1']: biasC['fc1'],
						 biasC_tf['fc2']: biasC['fc2'],
						 biasC_tf['out']: biasC['out'],
						 lamda_tf['fc1']: lamda['fc1'],
						 lamda_tf['fc2']: lamda['fc2'],
						 lamda_tf['out']: lamda['out'],
						 lamda_bias_tf['fc1']: lamda_bias['fc1'],
						 lamda_bias_tf['fc2']: lamda_bias['fc2'],
						 lamda_bias_tf['out']: lamda_bias['out']}
			train_L_step.run(feed_dict)
			
			if i % 100 == 0:
				# train_accuracy = accuracy.eval(
				# 	feed_dict = {x: X_batch, 
				# 		   		 y: y_batch})
				# print('step {}, training accuracy {}' .format(i, train_accuracy))
				train_loss, train_accuracy = \
				 		sess.run([loss_L_step, accuracy], feed_dict = feed_dict)
				index = j*num_epoch_in_each_L_train+epoch_train
				train_loss_L[index] = train_loss
				train_error_L[epoch] = 1 - train_accuracy
			# reference weight and bias
			w_bar = sess.run(W)
			bias_bar = sess.run(bias)
		######################################################################
		####################### accuracy using w #############################
		feed_dict = {	wC_tf['fc1']: wC_reshape['fc1'],
						wC_tf['fc2']: wC_reshape['fc2'],
						wC_tf['out']: wC_reshape['out'],
						biasC_tf['fc1']: biasC['fc1'],
						biasC_tf['fc2']: biasC['fc2'],
						biasC_tf['out']: biasC['out'],
						lamda_tf['fc1']: lamda['fc1'],
						lamda_tf['fc2']: lamda['fc2'],
						lamda_tf['out']: lamda['out'],
						lamda_bias_tf['fc1']: lamda_bias['fc1'],
						lamda_bias_tf['fc2']: lamda_bias['fc2'],
						lamda_bias_tf['out']: lamda_bias['out'],
						mu_tf: mu,
						x: data.validation.images, 
						y: data.validation.labels }
		val_loss, val_accuracy = \
		sess.run([loss_L_step, accuracy], feed_dict = feed_dict)
		val_loss_L[j] = val_loss
		val_error_L[j] = 1 - val_accuracy

		feed_dict = {	wC_tf['fc1']: wC_reshape['fc1'],
						wC_tf['fc2']: wC_reshape['fc2'],
						wC_tf['out']: wC_reshape['out'],
						biasC_tf['fc1']: biasC['fc1'],
						biasC_tf['fc2']: biasC['fc2'],
						biasC_tf['out']: biasC['out'],
						lamda_tf['fc1']: lamda['fc1'],
						lamda_tf['fc2']: lamda['fc2'],
						lamda_tf['out']: lamda['out'],
						lamda_bias_tf['fc1']: lamda_bias['fc1'],
						lamda_bias_tf['fc2']: lamda_bias['fc2'],
						lamda_bias_tf['out']: lamda_bias['out'],
						mu_tf: mu,
						x: data.test.images, 
						y: data.test.labels }
		test_loss, test_accuracy = \
		sess.run([loss_L_step, accuracy], feed_dict = feed_dict)
		test_loss_L[j] = test_loss
		test_error_L[j] = 1 - test_accuracy

		# print('epoch {} and test accuracy using w {}' .format(j, accuracy.eval(
		# 	feed_dict={x: data.validation.images, 
		# 			   y: data.validation.labels})))

		#######################################################################
		######## C Step #######################################################
		#######################################################################
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
			C[layer] = C[layer].reshape(-1)
		
		######################################################################
		####################### accuracy using wc ############################
		feed_dict = {   wC_tf['fc1']: wC_reshape['fc1'],
						wC_tf['fc2']: wC_reshape['fc2'],
						wC_tf['out']: wC_reshape['out'],
						biasC_tf['fc1']: biasC['fc1'],
						biasC_tf['fc2']: biasC['fc2'],
						biasC_tf['out']: biasC['out'],
						x: data.validation.images, 
						y: data.validation.labels }

		val_loss, val_accuracy = \
			sess.run([loss_compression, accuracy_compression], 
							feed_dict = feed_dict)

		feed_dict = {   wC_tf['fc1']: wC_reshape['fc1'],
						wC_tf['fc2']: wC_reshape['fc2'],
						wC_tf['out']: wC_reshape['out'],
						biasC_tf['fc1']: biasC['fc1'],
						biasC_tf['fc2']: biasC['fc2'],
						biasC_tf['out']: biasC['out'],
						x: data.test.images, 
						y: data.test.labels }

		val_loss_C[j] = val_loss
		val_error_C[j] = 1 - val_accuracy

		test_loss, test_accuracy = \
		sess.run([loss_compression, accuracy_compression], feed_dict = feed_dict)
		test_loss_C[j]= test_loss
		test_error_C[j] = 1 - test_accuracy

		# print('epoch {} and test accuracy using wc {}' 
		# 				.format(j, accuracy_compression.eval(
		# 					feed_dict={x: data.validation.images, 
		# 							   y: data.validation.labels,
		# 							   wC_tf['fc1']: wC_reshape['fc1'],
		# 							   wC_tf['fc2']: wC_reshape['fc2'],
		# 							   wC_tf['out']: wC_reshape['out'],
		# 							   biasC_tf['fc1']: biasC['fc1'],
		# 							   biasC_tf['fc2']: biasC['fc2'],
		# 							   biasC_tf['out']: biasC['out']})))
		#######################################################################
		############################ update lambda ############################
		for layer, _ in w_bar.items():
			lamda[layer] = lamda[layer] - mu * (w_bar[layer] - wC_reshape[layer])
			lamda_bias[layer] = lamda_bias[layer] - mu * (bias_bar[layer] - biasC[layer])

		norm_compression = 0
		for layer, _ in w_bar.items():
			norm_compression = LA.norm(w[layer] - wC[layer])

		print('norm of compression: {} ' .format(norm_compression) )

		if norm_compression < 0.001:
			break

	save_path = saver.save(sess, model_file_path)

###############################################################################
############################## LC WITH RETRAINING #############################
###############################################################################

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
model_file_path = './model/' + model_file_name 

with tf.Session() as sess:
	sess.run(init)
	feed_dict = {
		codebook_placeholder_tf['fc1']: C['fc1'],
		codebook_placeholder_tf['fc2']: C['fc2'],
		codebook_placeholder_tf['out']: C['out']
	}
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
			lr = learning_rate_decay ** j
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		
		feed_dict = {
			Z_W_int_tf['fc1']: Z_W_matrix['fc1'],
			Z_W_int_tf['fc2']: Z_W_matrix['fc2'],
			Z_W_int_tf['out']: Z_W_matrix['out'],
			Z_bias_int_tf['fc1']: Z_bias_matrix['fc1'],
			Z_bias_int_tf['fc2']: Z_bias_matrix['fc2'],
			Z_bias_int_tf['out']: Z_bias_matrix['out'],
			x: X_batch,
			y: y_batch,
			learning_rate: lr,
			momentum_tf: momentum
		}
		train_DC_ret_step.run(feed_dict = feed_dict)
		# if i % 100 == 0:
		# 	train_accuracy = accuracy.eval(
		# 		feed_dict={x: X_batch, 
		# 				   y: y_batch})
		# 	print('step {}, training accuracy {}' .format(i, train_accuracy))

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

		#train_loss_ref = sess.run(loss)
		
	save_path = saver.save(sess, model_file_path)
	# reference weight and bias
	C_LC_ret = sess.run(codebook_tf, feed_dict = feed_dict)


import dill
results_file_name = 'dill_global_variables_k_' + str(k) + '.pkl'
results_file_path = './results/' + results_file_name 
dill.dump_session(results_file_path)

# dill.load_session(results_file_path)


# with open(results_file_path, 'wb') as f:
# 	pickle.dump(df_ref,f)
# 	pickle.dump(df_DC,f)
# 	pickle.dump(df_DC_retrain,f)
# 	pickle.dump(df_LC,f)

# results_file_name = 'data_k_' + str(k)
# results_file_path = './results/' + data_file_name 

# with open(results_file_path, 'rb') as f:
# 	df_ref = pickle.load(f)
# 	df_DC = pickle.load(f)
# 	df_DC_retrain = pickle.load(f)
# 	df_LC = pickle.load(f)













