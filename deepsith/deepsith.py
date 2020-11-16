# Deep SITH
# PyTorch version 0.1.0
# Authors: Brandon G. Jacques and Per B. Sederberg

import torch
from torch import nn
from .isith import iSITH


from deepsith import iSITH
class _DeepSITH_core(nn.Module):
    def __init__(self, layer_params):
        super(_DeepSITH_core, self).__init__()

        hidden_size = layer_params.pop('hidden_size', layer_params['in_features'])
        in_features = layer_params.pop('in_features', None)
        batch_norm = layer_params.pop('batch_norm', True)
        act_func = layer_params.pop('act_func', None)
        self.batch_norm = batch_norm
        self.act_func = not (act_func is None)
        self.sith = iSITH(**layer_params)
        
        self.linear = nn.Linear(layer_params['ntau']*in_features,
                                hidden_size)
        nn.init.kaiming_normal_(self.linear.weight.data)
        
        if not (act_func is None):
            self.act_func = act_func
        if batch_norm:
            self.dense_bn = nn.BatchNorm1d(hidden_size)
            
    def forward(self, inp):
        # Outputs as : [Batch, features, tau, sequence]
        x = self.sith(inp)
        
        x = x.transpose(3,2).transpose(2,1)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.linear(x)
        if self.act_func:
            x = self.act_func(x)
        if self.batch_norm:
            x = x.transpose(2,1)
            x = self.dense_bn(x).transpose(2,1)
        return x

class DeepSITH(nn.Module):
    """A Module built for SITH like an LSTM

    Parameters
    ----------
    layer_params: list
        A list of dictionaries for each layer in the desired DeepSITH. All
        of the parameters needed for the SITH part of the Layers, as well as
        a hidden_size and optional act_func are required to be present.

    layer_params keys
    -----------------
    hidden_size: int (default in_features)
        The size of the output of the hidden layer. Please note that the
        in_features parameter for the next layer's SITH representation should be
        equal to the previous layer's hidden_size. This parameter will default
        to the in_features of the current SITH layer if not specified.
    act_func: torch.nn.Module (default None)
        The torch layer of the desired activation function, or None if no
        there is no desired activation function between layers.

    In addition to these keys, you must include all of the non-optional SITH
    layer keys in each dictionary. Please see the SITH docstring for
    suggestions.

    """
    def __init__(self, layer_params, dropout=.5):
        super(DeepSITH, self).__init__()
        self.layers = nn.ModuleList([_DeepSITH_core(layer_params[i])
                                      for i in range(len(layer_params))])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(len(layer_params) - 1)])
        
    def forward(self, inp):
        x = inp
        for i, l in enumerate(self.layers[:-1]):
            x = l(x)
            x = self.dropouts[i](x)
            x = x.unsqueeze(1).transpose(3,2)
        x = self.layers[-1](x)
        return x