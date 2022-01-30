#--------+---------+---------+---------+---------+---------+---------+---------+
#
# This file will be loaded as part of specific scripts and
# contains dependences on global variables introduced in these scripts
#
#   proj_path
#   data_path
#   model_path
#   plot_path
#   data_file_name
#
#   N_samples
#   train_size
#   valid_size
#
#   n_in
#
#--------+---------+---------+---------+---------+---------+---------+---------+


#--------+---------+---------+---------+---------+---------+---------+---------+
# Load data to pandas.DataFrame
#
import numpy as np
import pandas as pd
import os

current_path = os.path.abspath(os.getcwd())
print(f"Current path: '{current_path}'")

fn = os.path.join(proj_path, data_path, data_file_name)
print(f"Opening data file '{fn}'")

df_pmc = pd.read_csv(fn, sep=",", header=0)

#print(df_pmc.info())

#--------+---------+---------+---------+---------+---------+---------+---------+
# Remove all instances with 'log_posterior' = -inf
#

#print(df_pmc.groupby(np.isinf(df_pmc['log_posterior'])).count())
df_pmc = df_pmc.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
#print(df_pmc.groupby(np.isinf(df_pmc['log_posterior'])).count())

#--------+---------+---------+---------+---------+---------+---------+---------+
# Standardize/Normalize the data
#

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

std_scaler = StandardScaler()
nrm_scaler = MinMaxScaler()

df_std = pd.DataFrame(std_scaler.fit_transform(df_pmc))
df_nrm = pd.DataFrame(nrm_scaler.fit_transform(df_pmc))
df_std.columns = df_pmc.columns
df_nrm.columns = df_pmc.columns

#--------+---------+---------+---------+---------+---------+---------+---------+
# Split into training/validation/test data
#

from src.data_handling import data_to_numpy
from src.data_handling import split_data

X_all, Y_all = data_to_numpy(df_nrm)

if N_samples == 'All':
    N_samples = len(X_all)

X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_data(X_all[:N_samples], Y_all[:N_samples], train_size=train_size, valid_size=valid_size)

#--------+---------+---------+---------+---------+---------+---------+---------+
# Initialize Pytorch
#

import torch
from torch import nn, optim
from torch.nn.modules import Module

print(torch.__version__)

def get_torch_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

print(f"CUDA is available: {torch.cuda.is_available()}")
torch_device = get_torch_device()

# cpu seems to be faster than gpu
torch_device = 'cpu'
print(f"This computation can be run on '{torch_device}'")

#--------+---------+---------+---------+---------+---------+---------+---------+
# Definitions NN architectures
#

n_in = len(X_all[0])

arch_1 = [(2* n_in, nn.ReLU()), (1* n_in, nn.ReLU()), (1, None)]
arch_2 = [(2* n_in, nn.ReLU()), (2* n_in, nn.ReLU()), (1* n_in, nn.ReLU()), (1, None)]
arch_3 = [(3* n_in, nn.ReLU()), (2* n_in, nn.ReLU()), (1* n_in, nn.ReLU()), (1, None)]
arch_4 = [(2* n_in, nn.ReLU()), (2* n_in, nn.ReLU()), (2* n_in, nn.ReLU()), (2* n_in, nn.ReLU()), (1* n_in, nn.ReLU()), (1, None)]

arch_4sigmoid = [(2* n_in, nn.Sigmoid()), (2* n_in, nn.Sigmoid()), (2* n_in, nn.Sigmoid()), (2* n_in, nn.Sigmoid()), (1* n_in, nn.Sigmoid()), (1, None)]
arch_4lrelu = [(2* n_in, nn.LeakyReLU()), (2* n_in, nn.LeakyReLU()), (2* n_in, nn.LeakyReLU()), (2* n_in, nn.LeakyReLU()), (1* n_in, nn.LeakyReLU()), (1, None)]
arch_4prelu = [(2* n_in, nn.PReLU()), (2* n_in, nn.PReLU()), (2* n_in, nn.PReLU()), (2* n_in, nn.PReLU()), (1* n_in, nn.PReLU()), (1, None)]

#--------+---------+---------+---------+---------+---------+---------+---------+
# Hyper parameter settings
#

def hyper_SGD_MSE(l_r, n_e, b_s, mom):
    return {
        'learning rate' : l_r,
        'epochs'        : n_e,
        'batch size'    : b_s,
        'batch shuffle' : True,
        'optimizer'     : torch.optim.SGD,
        'momentum SGD'  : mom,
        'loss function' : torch.nn.MSELoss(),
        'project path'  : proj_path,
        'plot path'     : plot_path,
        'model path'    : model_path
    }

def hyper_Adagrad_MSE(l_r, n_e, b_s):
    return {
        'learning rate' : l_r,
        'epochs'        : n_e,
        'batch size'    : b_s,
        'batch shuffle' : True,
        'optimizer'     : torch.optim.Adagrad,
        'loss function' : torch.nn.MSELoss(),
        'project path'  : proj_path,
        'plot path'     : plot_path,
        'model path'    : model_path
    }

def hyper_Adadelta_MSE(l_r, n_e, b_s):
    return {
        'learning rate' : l_r,
        'epochs'        : n_e,
        'batch size'    : b_s,
        'batch shuffle' : True,
        'optimizer'     : torch.optim.Adadelta,
        'loss function' : torch.nn.MSELoss(),
        'project path'  : proj_path,
        'plot path'     : plot_path,
        'model path'    : model_path
    }

def hyper_Adam_MSE(l_r, n_e, b_s):
    return {
        'learning rate' : l_r,
        'epochs'        : n_e,
        'batch size'    : b_s,
        'batch shuffle' : True,
        'optimizer'     : torch.optim.Adam,
        'loss function' : torch.nn.MSELoss(),
        'project path'  : proj_path,
        'plot path'     : plot_path,
        'model path'    : model_path
    }

def hyper_AdamW_MSE(l_r, n_e, b_s):
    return {
        'learning rate' : l_r,
        'epochs'        : n_e,
        'batch size'    : b_s,
        'batch shuffle' : True,
        'optimizer'     : torch.optim.AdamW,
        'loss function' : torch.nn.MSELoss(),
        'project path'  : proj_path,
        'plot path'     : plot_path,
        'model path'    : model_path
    }