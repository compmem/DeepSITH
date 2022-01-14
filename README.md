<h2 align="center">
<a href="https://arxiv.org/abs/2104.04646">DeepSITH: Efficient Learning via Decomposition of
What and When Across Time Scales</a>
</h2>

<h4 align="center">
  <a href="#overview">Overview</a> |
  <a href="#installation">Installation</a> |
  <a href="#deepsith-use">DeepSITH</a> |
  <a href="#examples">Examples</a>  
</h4>


## Overview

![DeepSITHLayout](/figures/model_config.png)

Here, we introduce DeepSITH, a network comprising biologically-inspired Scale-Invariant Temporal History (SITH) modules in series with dense connections between layers. SITH modules respond to their inputs with a geometrically-spaced set of time constants, enabling the DeepSITH network to learn problems along a continuum of time-scales.

## Installation

The easiest way to install the DeepSITH module is with pip.

    pip install .
    
### Requirements

DeepSITH requires at least PyTorch 1.8.1. It works with cuda, so please follow the instructions for installing pytorch and cuda <a href="https://pytorch.org/get-started/locally/">here</a>.

## DeepSITH
DeepSITH is a pytorch module implementing the neurally inspired SITH representation of working memory for use in neural networks. The paper outlining the work detailed in this repository was published at NeurIPS 2021 <a href="https://proceedings.neurips.cc/paper/2021/hash/e7dfca01f394755c11f853602cb2608a-Abstract.html">here</a>. 

Jacques, B., Tiganj, Z., Howard, M., &amp; Sederberg, P. (2021, December 6). DeepSITH: Efficient learning via decomposition of what and when across Time Scales. Advances in Neural Information Processing Systems. 

Primarily, this module utilizes SITH, the Scale-Invariant Temporal History, representation. With SITH, we are able to compress the history of a time series in the same way that human working memory might. For more information, please refer to the paper. 

### DeepSITH use

The DeepSITH module in pytorch will initialize as a series of deepsith layers, parameterized by the argument `layer_params`, which is a list of dictionaries. Below is an example initializing a 2 layer DeepSITH module, where the input time-series only has 1 feature. 

    from deepsith import DeepSITH
    from torch import nn as nn
    
    # Tensor Type. Use torch.cuda.FloatTensor to put all SITH math 
    # on the GPU.
    ttype = torch.FloatTensor
    
    sith_params1 = {"in_features":1, 
                    "tau_min":1, "tau_max":25.0, 'buff_max':40,
                    "k":84, 'dt':1, "ntau":15, 'g':.0,  
                    "ttype":ttype, 'batch_norm':True,
                    "hidden_size":35, "act_func":nn.ReLU()}
    sith_params2 = {"in_features":sith_params1['hidden_size'], 
                    "tau_min":1, "tau_max":100.0, 'buff_max':175,
                    "k":40, 'dt':1, "ntau":15, 'g':.0, 
                    "ttype":ttype, 'batch_norm':True,
                    "hidden_size":35, "act_func":nn.ReLU()}
    lp = [sith_params1, sith_params2]
    deepsith_layers = DeepSITH(layer_params=lp, dropout=0.2)

Here, we have the first layer only having 15 taustar from `tau_min=1.0` to `tau_max=25`. The second layer is set up to go from `1.0` to `100.0`, which gives it 4 times the temporal range. We found that the logarithmic increase of layer sizes to work well for the experiments in this repository. 

The DeepSITH module expects an input signal of size (batch_size, 1, sith_params1["in_features"], Time). 

If you want to use **only** the SITH module, which is a part of any DeepSITH layer, you can initialize a SITH using the following parameters. Note, these parameters are also used in the dictionaries above.

#### SITH Parameters
- tau_min: float
    The center of the temporal receptive field for the first taustar produced. 
- tau_max: float
    The center of the temporal receptive field for the last taustar produced. 
- buff_max: int
    The maximum time in which the filters go into the past. NOTE: In order to 
    achieve as few edge effects as possible, buff_max needs to be bigger than
    tau_max, and dependent on k, such that the filters have enough time to reach 
    very close to 0.0. Plot the filters and you will see them go to 0. 
- k: int
    Temporal Specificity of the taustars. If this number is high, then taustars
    will always be more narrow.
- ntau: int
    Number of taustars produced, spread out logarithmically.
- dt: float
    The time delta of the model. There will be int(buff_max/dt) filters per
    taustar. Essentially this is the base rate of information being presented to the model
- g: float
    Typically between 0 and 1. This parameter is the scaling factor of the output
    of the module. If set to 1, the output amplitude for a delta function will be
    identical through time. If set to 0, the amplitude will decay into the past, 
    getting smaller and smaller. This value should be picked on an application to 
    application basis.
- ttype: Torch Tensor
    This is the type we set the internal mechanism of the model to before running. 
    In order to calculate the filters, we must use a DoubleTensor, but this is no 
    longer necessary after they are calculated. By default we set the filters to 
    be FloatTensors. NOTE: If you plan to use CUDA, you need to pass in a 
    cuda.FloatTensor as the ttype, as using .cuda() will not put these filters on 
    the gpu. 

Initializing SITH will generate several attributes that depend heavily on the values of the parameters. 

- c: float
    `c = (tau_max/tau_min)**(1./(ntau-1))-1`. This is the description of how the distance between
    taustars evolves. 
- tau_star: DoubleTensor
    `tau_star = tau_min*(1+c)**torch.arange(ntau)`. This is the array filled with all of the
    centers of all the tau_star receptive fields. 
- filters: ttype
    The generated convolutional filters to generate SITH output. Will be applied as a convolution
    to the input time-series.

Importantly, this module should be socketed into a larger pytorch model, where the final layers are transforming the last output of the **SITH->Dense Layer** into the shape of the output required for a particular task. We, for instance, use a DeepSITH_Classifier model for most of the tasks within this repository. 

    class DeepSITH_Classifier(nn.Module):
        def __init__(self, out_features, layer_params, dropout=.5):
            super(DeepSITH_Classifier, self).__init__()
            last_hidden = layer_params[-1]['hidden_size']
            self.hs = DeepSITH(layer_params=layer_params, dropout=dropout)
            self.to_out = nn.Linear(last_hidden, out_features)
        def forward(self, inp):
            x = self.hs(inp)
            x = self.to_out(x)
            return x

## Examples

In the `experiments` folder are the experiments that were included in the paper. Everything to recreate the results therein is included. Everything is in jupyter notebooks. We have also included everything needed to recreate the figures from the paper, but with your results if you change file names around. 


