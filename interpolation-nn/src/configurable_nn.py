
import torch.nn as nn

class Configurable_Linear_NN(nn.Module):
    """
    Neural Network with fully-connected forward architecture
    (for extrapolation of Log Posterior probability)
    """
    def __init__(self, n_in, layers_descr):
        """Constructor

        :param n_in: number of input features
        :type  n_in: int

        :param layers_descr: list of tupels: (number of neurons, activation function)  
                             use activation function = None for no activation function
        """
        super().__init__()
        
        self.layers = nn.ModuleList()  # holds layers
        self.n_in = n_in               # maybe for later use

        for n_out, act_fn in layers_descr:
            self.layers.append(nn.Linear(n_in, n_out))
            n_in = n_out  # for the next layer
            if act_fn is not None:
                assert isinstance(act_fn, nn.Module), \
                    "Each tuple should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(act_fn)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x