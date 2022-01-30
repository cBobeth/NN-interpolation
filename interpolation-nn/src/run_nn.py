import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm
from tqdm import tqdm

import matplotlib
from matplotlib import rcParams
try:
    if __IPYTHON__:
        pass
except NameError as e:
    matplotlib.use('pgf')

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
    ]),
})

# set some default values for plotting
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Sans serif'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['font.weight'] = 'normal'

matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['axes.titlepad'] = 12
matplotlib.rcParams['axes.formatter.use_mathtext'] = True

matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

matplotlib.rcParams['text.usetex'] = True

matplotlib.rcParams['pgf.preamble'] = r'''
\usepackage{amsmath}
\usepackage{xcolor}
'''

matplotlib.rcParams['errorbar.capsize'] = 5

# Some example dictionary with hyper parameters
#
example_hyper_dict = {
    'learning rate' : 0.03,
    'epochs'        : 1000,
    'batch size'    : 64,
    'batch shuffle' : True,
    'optimizer'     : torch.optim.SGD,
    'momentum SGD'  : 0.1,
    'loss function' : torch.nn.MSELoss(),
    'project path'  : "proj_path",
    'plot path'     : "plot_path",
    'model path'    : "model_path"
}


class Run_Model:
    """
    Represents a single run of a model + hyperparameter setting, which comprises
    - training run
    - test run
    - storing the training history (training loss, validation loss), error counts
    """
    def __init__(self, tag, model, n_in, model_arch, hyper_dict, data, torch_device='cpu', verbose=True):
        """
        Constructor of the run, starts training and scoring

        :param tag: some identifier of the run (used for file names)
        :type  tag: string

        :param model: some model class of the model to be used
        :type  model: for the moment only 'Configurable_Linear_NN'

        :param n_in: number of input features
        :type  n_in: int

        :param model_arch: model architecture: list of tupels: (number of neurons, activation function)  
                           use activation function = None for no activation function

        :param hyper_dict: dictionary with values for hyper parameters
        :type  hyper_dict: KEY:VALUE pairs: 'learning rate', 'epochs', 'batch size', 'optimizer', 'loss function'

        :param data: the training, validation and test data in the form of list/tuple with 6 entries [x_train, y_train, x_valid, y_valid, x_test, y_test]
        :type  data: numpy arrays

        :param torch_device: a valid name of a torch device for calculations
        :type  torch_device: string, default = 'cpu'
        """
        self.verbose = verbose                   # Verbose messages
        self.tag = tag
        self.model = model(n_in, model_arch)
        self.hyper_dict = hyper_dict
        self.data = data
        self.torch_device = torch_device

        # intervals for ratios r = y_predicted/y_true
        # to test accuracies of validation and test data,
        # for example [0.99, 1.01] means a deviation of -1% to +1% etc. 
        self.error_ranges = [(0.99, 1.01), (0.9, 1.1), (0.5, 2.0)]
        self.loss_hst = []
        self.error_hst = []
        self.outlier_lst = []
        
        if self.verbose:
            print(f"\n\n")
            print(f"[Run_model] '{self.tag}' on torch_device = {self.torch_device}")
            print(f"-->          Model = {model}")
            print(f"-->   Architecture = {model_arch}")
            print(f"--> Hyperparamters = ")
            max_len = max([len(x) for x in self.hyper_dict.keys()])
            for key in self.hyper_dict.keys():
                print(f"         {key:{max_len}s} = {self.hyper_dict[key]}")

        # call training run and testing
        self.train()
        self.test()


    def _data_iter(self, batch_size, X, Y, shuffle=True):
        """
        Iterator over batches of features X and labels Y.
        X and Y should have same length.
    
        :param batch_size: size of batch
        :type  bathc_size: int
    
        :param X: torch tensor with features
        
        :param Y: torch tensor with labels
    
        :param shuffle: if True, shuffle randomly the data before splitting into batches
        :type  shuffle: bool
        
        :return: batch as a tupel of (X_batch, Y_batch) batched data, torch tensors, on same device as X, Y

        ..note::
            To enable stochastic gradient descent based optimizers need to use shuffle=True
        """
        len_data = len(X)
        idx = list(range(len_data))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len_data, batch_size):
            batch_idx = idx[i: min(i + batch_size, len_data)]
            yield X[batch_idx], Y[batch_idx]


    def train(self, reset_history=True):
        """
        Launch training of the model for the setting of hyperparameters in hyper_dict.
        Stores a history of training and validation losses for epochs.

        :param reset_history: if True, the loss history will be reset, otherwise
        :type  reset_history: bool

        ..note::
            Use reset_history=False when continuing training (ToDo: not yet implemented completely)
        """
        l_r = self.hyper_dict['learning rate']
        n_epochs = self.hyper_dict['epochs']
        batch_size = self.hyper_dict['batch size']
        if 'batch shuffle' in self.hyper_dict:
            batch_shuffle = self.hyper_dict['batch shuffle']
        else:
            batch_shuffle = True

        optimizer = self.hyper_dict['optimizer']
        loss_fn   = self.hyper_dict['loss function']

        # move model to device
        # => do it before optimizer with model parameters is instantiated
        self.model.to(self.torch_device)

        # initialize optimizer with hyper parameters
        if optimizer == torch.optim.SGD:
            if 'momentum SGD' in self.hyper_dict:
                mom = self.hyper_dict['momentum SGD']
            else:
                mom = 0.0
            optim = optimizer(self.model.parameters(), lr=l_r, momentum=mom)
        else:
            optim = optimizer(self.model.parameters(), lr=l_r)

        # move data to device before training
        # => data_iter will batch on correct device, avoids copying batches between devices
        x_tr  = torch.from_numpy(self.data[0]).to(self.torch_device)
        y_tr  = torch.from_numpy(self.data[1]).to(self.torch_device)
        x_val = torch.from_numpy(self.data[2]).to(self.torch_device)
        y_val = torch.from_numpy(self.data[3]).to(self.torch_device)
        
        # enable also continutation of run
        if reset_history:
            self.loss_hst = []
            self.error_hst = []

        len_train = len(self.data[0])     # length of training data
        len_valid = len(self.data[2])

        if self.verbose:
            print(f"[Run_Model.train()] training data length = {len_train}, validation data length = {len_valid}")

        for epoch in tqdm(range(n_epochs)):
            # training part
            # put model into "training mode" as oposed to "prediction mode"
            self.model.train()  # optional when not using model specific layer
    
            train_loss = 0.0        
            for inputs, targets in self._data_iter(batch_size, x_tr, y_tr, shuffle=batch_shuffle):
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets.reshape((-1,1)))
                loss.backward()
                optim.step()
                #
                loss_v = loss.item()
                # here inputs.size(0) undoes the mean taken in torch.lossMSE() for each batch
                train_loss += loss_v * inputs.size(0)
                #
                optim.zero_grad(set_to_none=False)  # =True should give some speed up, but different behavior depending on grad is zero or None
            # take eventually the mean w.r.t. whole data set
            train_loss /= len_train
    
            # validation part
            self.model.eval()  # optional when not using model specific layer
            
            valid_loss = 0.0
            error_cnts = np.zeros(len(self.error_ranges))
            for inputs, targets in self._data_iter(batch_size, x_val, y_val, shuffle=batch_shuffle):
                output = self.model(inputs)
                loss = loss_fn(output, targets.reshape((-1,1)))
                valid_loss += loss.data.item() * inputs.size(0)
                # increase counters for error_ranges for predictions in this batch
                for o_pr, o_tr in zip(output, targets.reshape((-1,1))):  # pr = predicted, tr = true
                    if o_tr == 0.:
                        r = 2.1  # choose some value outside of largest error range
                    else:
                        r = o_pr/o_tr
                    for ix, rng in enumerate(self.error_ranges):
                        if (rng[0] <= r and r <= rng[1]):
                            error_cnts[ix] += 1
            valid_loss /= len_valid
            error_cnts /= len_valid

            # append loss and error counts of this epoch to history
            self.loss_hst.append([train_loss, valid_loss])
            self.error_hst.append(error_cnts)

        print(f"Training summary:\n"\
              f"    Training   loss = {self.loss_hst[-1][0]:.5f}\n"\
              f"    Validation loss = {self.loss_hst[-1][1]:.5f}\n"\
              f"       Error counts = " 
              + str(['%3.1f' % (100.* x) for x in self.error_hst[-1]])
              +" %")    


    def test(self, ):
        """
        Predict the score on test data after training 
        ToDo: implement
              Could pass the test data in the tuple as 5th and 6th entries
              'data' = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test) 
        """
        self.model.eval()  # optional when not using model specific layer

        batch_size = self.hyper_dict['batch size']
        if 'batch shuffle' in self.hyper_dict:
            batch_shuffle = self.hyper_dict['batch shuffle']
        else:
            batch_shuffle = True

        loss_fn   = self.hyper_dict['loss function']

        # move data to device before training
        # => data_iter will batch on correct device, avoids copying batches between devices
        x_test = torch.from_numpy(self.data[4]).to(self.torch_device)
        y_test = torch.from_numpy(self.data[5]).to(self.torch_device)

        len_test = len(self.data[4])
        if self.verbose:
            print(f"[Run_Model.test()] test data length = {len_test}")

        # determine for test data
        # - loss
        # - error_counts
        # - outliers
        test_loss = 0.0
        error_cnts = np.zeros(len(self.error_ranges))
        outliers = []
        for inputs, targets in self._data_iter(batch_size, x_test, y_test, shuffle=batch_shuffle):
            output = self.model(inputs)
            loss = loss_fn(output, targets.reshape((-1,1)))
            test_loss += loss.data.item() * inputs.size(0)
            # increase counters for error_ranges for predictions in this batch
            for o_pr, o_tr in zip(output.detach().numpy(), targets.reshape((-1,1)).detach().numpy()):  # pr = predicted, tr = true
                o_pr = o_pr[0]
                o_tr = o_tr[0]
                if o_tr == 0.:
                    r = 2.1  # choose some value outside of largest error range
                else:
                    r = o_pr/o_tr
                for ix, rng in enumerate(self.error_ranges):
                    if (rng[0] <= r and r <= rng[1]):
                        error_cnts[ix] += 1
                if (r <= self.error_ranges[-1][0] or self.error_ranges[-1][1] <= r):
                    outliers.append([o_pr, o_tr])
        test_loss /= len_test
        self.outlier_lst = outliers
        error_cnts /= len_test

        print(f"Test summary:\n"\
              f"          Test loss = {test_loss:.5f}\n"\
              f"       Error coutns = "
              + str(['%3.1f' % (100.* x) for x in error_cnts])
              +" %"
             )


    def predict(self, X):
        """
        Predict y = model(X) based on the trained model

        :param X: input features
        :type  X: torch tensor

        ..note::
            Uses global variable 'torch_device'
        """
        self.model.eval()
        return self.model(X.to(self.torch_device))


    def plot_loss_hst(self, figsize_x=20, figsize_y=4, y_scale='log', y_max='None'):
        """
        Plot history of training and validation loss

        :param figsize_x, figsize_y: x- and y-sizes of the figure
        :type  figsize_x, figsize_y: int

        :param y_scale: log or linear scale for y-axis
        :type  y_scale: 'linear, 'log'

        :param y_max: max value of y-axis
        :type  y_max: float

        :return: none

        ..note::
            Saves plot as 'loss_hst_{tag}.pdf' file in 'plot path'.
        """
        if y_max == 'None':
            y_max = max(self.loss_hst[:][0])
        
        y_min = 0.8* min(min(self.loss_hst))  # 80% below smallest value

        plt.rcParams['figure.figsize'] = [figsize_x, figsize_y]

        fig, ax = plt.subplots()
        ax.set_yscale(y_scale)
        ax.set_ylim(y_min, y_max)
        ax.set(xlabel=r'$N_{\mathrm{epochs}}$')
        if y_scale == 'log':
            ax.set(ylabel=r'log(MSE Loss)')
        else:
            ax.set(ylabel=r'MSE Loss')

        loss_hst = np.array(self.loss_hst).transpose()
        label_lst = [r'training data', r'validation data']
        for loss, lab in zip(loss_hst, label_lst):  
            ax.plot(loss, label=lab)

        ax.legend(loc = 'lower left')

        proj_path = self.hyper_dict['project path']
        plot_path = self.hyper_dict['plot path']
        data_file_name = 'loss_hst-' + self.tag + '.pdf'
        fname = fn = os.path.join(proj_path, plot_path, data_file_name)
        
        plt.savefig(fname, format='pdf')


    def plot_error_hst(self, figsize_x=20, figsize_y=4, y_max=0.5):
        """
        Plot history of error counts in the error ranges found for the validation data set
        on log-scale for y-axis.

        :param figsize_x, figsize_y: x- and y-sizes of the figure
        :type  figsize_x, figsize_y: int

        :param y_max: max value of y-axis
        :type  y_max: float

        :return: none

        ..note::
            Saves plot as 'error_hst_{tag}.pdf' file in 'plot path'.
            y_scale = 'linear' does not make sense here.
        """
        plt.rcParams['figure.figsize'] = [figsize_x, figsize_y]
        
        fig, ax = plt.subplots()

        error_hst = 1.- np.array(self.error_hst)  
        y_min = 0.8* np.min(error_hst)  # 80% below smallest value
        y_min = 5e-4
        for ix, rng in enumerate(self.error_ranges):  
            ax.plot(error_hst[:, ix], label=f"error range = {rng}")
        
        y_scale = 'log'
        ax.set_yscale(y_scale)
        ax.set_ylim(y_min, y_max)
        ax.set(xlabel=r'$N_{\mathrm{epochs}}$')
        ax.set(ylabel=r'$\log(1 - \mathrm{percentage})$')

        plt.axhline(y=1e-1, color='green', linestyle='dotted')
        plt.axhline(y=5e-2, color='green', linestyle='dashdot')
        plt.axhline(y=1e-2, color='green', linestyle='dashed')
        plt.axhline(y=1e-3, color='green', linestyle='-')
        # assumes there are at least 10 epochs
        n_e = len(self.error_hst) - 10
        plt.text(n_e, 1e-1, r'$90\%$', fontsize=10, va='center', ha='center', backgroundcolor='w')
        plt.text(n_e, 5e-2, r'$95\%$', fontsize=10, va='center', ha='center', backgroundcolor='w')
        plt.text(n_e, 1e-2, r'$99\%$', fontsize=10, va='center', ha='center', backgroundcolor='w')
        plt.text(n_e, 1e-3, r'$99.9\%$', fontsize=10, va='center', ha='center', backgroundcolor='w')
        
        ax.legend(loc = 'lower left')

        proj_path = self.hyper_dict['project path']
        plot_path = self.hyper_dict['plot path']
        data_file_name = 'error_hst-' + self.tag + '.pdf'
        fname = fn = os.path.join(proj_path, plot_path, data_file_name)
        
        plt.savefig(fname, format='pdf')


    def plot_outliers(self, figsize_x=5, figsize_y=5):
        """
        Scatter plot of outlier distribution y_predicted vs. y_true.
        Show range between [0, 1] assuming that log-posterior is normalized to interval [0, 1].

        :param figsize_x, figsize_y: x- and y-sizes of the figure
        :type  figsize_x, figsize_y: int

        :return: none

        ..note::
            Saves plot as 'outliers_{tag}.pdf' file in 'plot path'.
        """
        plt.rcParams['figure.figsize'] = [figsize_x, figsize_y]
        
        fig, ax = plt.subplots()

        out_lst = np.array(self.outlier_lst)

        ax.scatter(out_lst[:, 1], out_lst[:, 0])

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        ax.set(xlabel=r'$y_{\mathrm{true}}$')
        ax.set(ylabel=r'$y_{\mathrm{predicted}}$')

        x_val = np.linspace(0.0, 1.0, 10)

        # plot error ranges
        plt.fill_between(x_val, self.error_ranges[2][1]* x_val, self.error_ranges[1][1]* x_val, color='red', alpha=0.1)
        plt.fill_between(x_val, self.error_ranges[1][1]* x_val, self.error_ranges[0][1]* x_val, color='blue', alpha=0.1)
        plt.fill_between(x_val, self.error_ranges[0][1]* x_val, self.error_ranges[0][0]* x_val, color='green', alpha=0.1)
        plt.fill_between(x_val, self.error_ranges[0][0]* x_val, self.error_ranges[1][0]* x_val, color='blue', alpha=0.1)
        plt.fill_between(x_val, self.error_ranges[1][0]* x_val, self.error_ranges[2][0]* x_val, color='red', alpha=0.1)
        ax.plot(x_val, self.error_ranges[0][0]* x_val, color='green', linestyle='-')
        ax.plot(x_val, self.error_ranges[0][1]* x_val, color='green', linestyle='-')
        ax.plot(x_val, self.error_ranges[1][0]* x_val, color='blue', linestyle='--')
        ax.plot(x_val, self.error_ranges[1][1]* x_val, color='blue', linestyle='--')
        ax.plot(x_val, self.error_ranges[2][0]* x_val, color='red', linestyle='dotted')
        ax.plot(x_val, self.error_ranges[2][1]* x_val, color='red', linestyle='dotted')

        proj_path = self.hyper_dict['project path']
        plot_path = self.hyper_dict['plot path']
        data_file_name = 'outlier-' + self.tag + '.pdf'
        fname = fn = os.path.join(proj_path, plot_path, data_file_name)
        
        plt.savefig(fname, format='pdf')