import argparse
import os
import numpy as np
import pickle
# Set up argument parser
ap = argparse.ArgumentParser()
# Single positional argument, nargs makes it optional
ap.add_argument('k', nargs='?', default=2)
# codebook size
k = int(ap.parse_args().k)

import pickle
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

import matplotlib.pyplot as plt
plots_folder = './plots'
file_name = 'destination_path.eps'
file_path = plots_folder + file_name

plt.savefig(file_path, format='eps', dpi=300)