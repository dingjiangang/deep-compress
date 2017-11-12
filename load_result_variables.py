import argparse
import os
import numpy as np
import pickle
# Set up argument parser
ap = argparse.ArgumentParser()
# Single positional argument, nargs makes it optional
ap.add_argument('kk', nargs='?', default=2)
# codebook size
kk = int(ap.parse_args().kk)

import dill
results_file_name = 'dill_global_variables_k_' + str(kk) + '.pkl'
results_file_path = './results/' + results_file_name 

dill.load_session(results_file_path)
print(val_loss_ref)
