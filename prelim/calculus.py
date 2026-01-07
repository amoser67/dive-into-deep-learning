# %matplotlib inline
import numpy as np
from matplotlib_inline import backend_inline
import matplotlib
import matplotlib.pyplot as plt
from d2l import torch as d2l
import torch


def f(x):
    # return 3 * x**2 - 4*x
    return x**3 - 1/x

# x = np.linspace(0, 2 * np.pi, 200)
# y = np.sin(x)
#

# fig, ax = plt.subplots()
# ax.plot(x, y)

# plt.savefig('calculus.png')
# plt.show(block=True)

x = np.arange(-5, 5, .2)
# d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# d2l.plt.show()
# print(x)

d2l.plot(
    x,
    [f(x), 2*x - 2],
    'x',
    'f(x)',
    legend=['f(x)', 'Tangent line (x=1)']
)
d2l.plt.show()


"""
Auto Differentiation
"""

x = torch.arange(4.0)
# print(x)
# tensor([0., 1., 2., 3.])

x.requires_grad_(True)
# print(x.grad)
# None

y = 2 * torch.dot(x, x)
# print(y)
# tensor(28., grad_fn=<MulBackward0>)

y.backward()
# print(x.grad)
# tensor([ 0.,  4.,  8., 12.])

# print(x.grad == 4 * x)
# tensor([True, True, True, True])

x.grad.zero_()  # Reset gradient buffer.
y = x.sum()
# print(y)
# tensor(6., grad_fn=<SumBackward0>)

y.backward()
# print(x.grad)
# tensor([1., 1., 1., 1.])

x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
# print(x.grad)
# tensor([0., 2., 4., 6.])



