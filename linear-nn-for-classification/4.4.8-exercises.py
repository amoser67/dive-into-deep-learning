import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# dfg
a = 1

"""
Question: INT8 is a very limited format consisting of nonzero numbers from 1 to 255.
How could you extend its dynamic range without using more bits?
Do standard multiplication and addition still work?

You could use log(int8) or exp(int8), or create a tensor of INT8 values that represents the digits in a larger
number, or some other relationship between the components (mult/add). In any case, multiplication and addition
operations would need to be defined differently.
"""