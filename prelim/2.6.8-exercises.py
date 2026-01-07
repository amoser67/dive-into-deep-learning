import random
import torch
from torch.distributions.multinomial import Multinomial
from torch.distributions.normal import Normal
from d2l import torch as d2l

"""
4. Draw m samples from distribution with mean 0 and unit variance. Find the averages for each value of m.

    num_draws = 100
    prob_dist = Normal(0, 1)
    results = prob_dist.sample((num_draws,))
    print(results)
    cum_sum = results.cumsum(dim=0)
    cum_num_draws = torch.arange(1, num_draws + 1, dtype=torch.float)
    averages = torch.div(cum_sum, cum_num_draws)
    print(averages)
"""


