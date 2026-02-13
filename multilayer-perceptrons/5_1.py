import torch
from d2l import torch as d2l


"""
Exercise: Compute the derivative of the pReLU activation function.

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.prelu(x, torch.tensor([3.0]))
d2l.plot(x.detach(), y.detach(), xlabel='x', ylabel='prelu(x)', figsize=(5, 2.5))
d2l.plt.show()

"""
