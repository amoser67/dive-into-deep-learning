import random
import torch
from torch.distributions.multinomial import Multinomial
from torch.distributions.normal import Normal
from d2l import torch as d2l


num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print(f"heads: {heads}, tails: {tails}")


fair_probs = torch.tensor([0.5, 0.5])

# 100 tosses
print(Multinomial(100, fair_probs).sample())
# tensor([54., 46.])
# print(Multinomial(100, fair_probs).sample() / 100)
# tensor([0.5400, 0.4600])

# 10,000 tosses
# print(Multinomial(10_000, fair_probs).sample())
# tensor([4999., 5001.])
# print(Multinomial(10_000, fair_probs).sample() / 10_000)
# tensor([0.4986, 0.5014])


"""
The Law of Large Numbers

The average of the results obtained from a large number of independent
random samples converges to the the true value, if it exists.

More formally:
Given a sample of independent and identically distributed values,
the sample mean converges to the true mean.
"""

"""
Central Limit Theorem

Under appropriate conditions, the distribution of a normalized version
of the sample mean converges to a standard normal distribution.
"""


counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
# print(cum_counts)
# tensor([[1., 0.],
#         [1., 1.],
#         [1., 2.],
#         [2., 2.],
#         [2., 3.],
#         ...])
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# print(estimates)
# tensor([[1.0000, 0.0000],
#         [0.5000, 0.5000],
#         [0.3333, 0.6667],
#         [0.5000, 0.5000],
#         [0.4000, 0.6000]
#         ...])
estimates = estimates.numpy()
# print(estimates)
# [[1.         0.        ]
#  [0.5        0.5       ]
#  [0.33333334 0.6666667 ]
#  [0.5        0.5       ]
#  [0.4        0.6       ]
#  ...]

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()

"""
Exercises
"""

"""
1. 
"""

