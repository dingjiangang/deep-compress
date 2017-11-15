import argparse
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

train_loss_ref = {}
train_error_ref = {}
val_loss_ref = {}
val_error_ref = {}
test_loss_ref = {}
test_error_ref = {}

total_minibatches = 100000
num_minibathes_in_epoch = 105

iter_ref = {}
label_ref = {}
for k in [2,4,8,16,32,64]: 
	label_ref[k] = 'k = ' + str(k)
	iter_ref[k] = np.arange(0,total_minibatches,num_minibathes_in_epoch) 


for k in [2,4,8,16,32,64]:
	file_pickle = './results/only_ref_results_pickle_k_' + str(k) + '.pkl'
	with open(file_pickle,'rb') as f:
		df_ref = pickle.load(f)
	train_loss_ref[k] = df_ref['train_loss_ref']
	train_error_ref[k] = df_ref['train_error_ref']
	val_loss_ref[k] = df_ref['val_loss_ref']
	val_error_ref[k] = df_ref['val_error_ref']
	test_loss_ref[k] = df_ref['test_loss_ref']
	test_error_ref[k] = df_ref['test_error_ref']

	file_pickle = './results/results_pickle_k_' + str(k) + '.pkl'
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



# import matplotlib.pyplot as plt
plots_folder = './plots'

# file_name = 'destination_path.eps'
# file_path = plots_folder + file_name

# import matplotlib.pyplot as plt
# import numpy as np

# num_plots = 20

# # Have a look at the colormaps here and decide which one you'd like:
# # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
# colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

# # Plot several different functions...
# x = np.arange(10)
labels = []
fig, ax = plt.subplots()
for i in [2,4,8,16,32,64]:
    ax.plot(iter_ref[k],train_loss_ref[k])
    labels.append(r'$k = + {}$' .format(k) )
legend = ax.legend(loc='upper center', shadow=True)

plt.show()

# # I'm basically just demonstrating several different legend options here...
# plt.legend(labels, ncol=4, loc='upper center', 
#            bbox_to_anchor=[0.5, 1.1], 
#            columnspacing=1.0, labelspacing=0.0,
#            handletextpad=0.0, handlelength=1.5,
#            fancybox=True, shadow=True)

# plt.show()

# plt.plot()
# plt.savefig(file_path, format='eps', dpi=300)