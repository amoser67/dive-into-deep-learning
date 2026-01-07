import torch

"""
We are interested in differentiating the function y = 2x^T * x with respect to column vector x.
"""

# First, we assign x an initial value.
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
y.backward()
# print(x.grad)
# tensor([2., 2., 2., 2.])
# D2L Comment: My understanding about 2.5.1’s pytorch code blocks:
# y = x.sum() is just a way of defining a computational graph in PyTorch, which means evaluating each component of x
# and adding them up. Gradients are computed on each component of x, NOT on the y graph. Evaluating gradient on a
# component of x means computing for y_i = x_i (which yields 1).
#
# The same principal applies to the y = 2 * x^Tx part, y is NOT 2x^2, it’s a computational graph for evaluating
# 2 * x^Tx where each component of it is actually y_i = 2 * x_i * x_i. So the graph y is in fact sum { 2 * x_i * x_i }.


x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
# print(x.grad)
# tensor([0., 2., 4., 6.])


"""
Exercises
"""

# Why is the second derivative much more expensive to compute than the second derivative?
# Comment
# Assuming we’re talking about the derivative of a scalar with respect to a tensor, the first derivative will be of the
# same size as the tensor. But the second derivative will be of that size, squared! Further, autodiff only requires a
# single backward pass to find the gradients - but if we want to find the full Hessian, we then need to find the
# derivative of each component of our initial gradients. So you’d have to a full backwards pass for every parameter,
# or find a more efficient way of computing this.

# 2
# Q: What happens if you try to run the "backward" method a second time?
# A: Error: Saved intermediate values of the graph are freed when you call .backward() or autograd.grad().

def g(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2

    # print("b")
    # print(b)
    # tensor([[810.3671],
    #         [987.4021],
    #         [5.7273]], grad_fn= < MulBackward0 >)

    # print(b.sum())
    # tensor(1803.4966, grad_fn= < SumBackward0 >)

    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(3, 1), requires_grad=True)
print(a)
# tensor([[-1.8164],
#         [-0.4624],
#         [ 0.7512]], requires_grad=True)

d = g(a)
print(d)
# tensor([[-92999.7656],
#         [-23674.4883],
#         [ 38461.3906]], grad_fn=<MulBackward0>)

print(d.sum())
# tensor(-78212.8594, grad_fn=<SumBackward0>)

d.sum().backward()
print(a.grad)
# tensor([[51200.],
#         [51200.],
#         [51200.]])

print(a.grad == d / a)
# tensor([[True],
#         [True],
#         [False]])


# tensor([[ 0.7278],
#         [ 0.5848],
#         [-1.2463]], requires_grad=True)
# tensor([[  745.2277],
#         [  598.8735],
#         [-1276.2585]], grad_fn=<MulBackward0>)
# tensor(67.8428, grad_fn=<SumBackward0>)
# tensor([[1024.],
#         [1024.],
#         [1024.]])
# tensor([[True],
#         [True],
#         [True]])

# tensor([[-1.4125],
#         [ 0.2388],
#         [-0.8448]], requires_grad=True)
# tensor([[-144643.1562],
#         [  24455.0898],
#         [ -86507.7188]], grad_fn=<MulBackward0>)
# tensor(-206695.7812, grad_fn=<SumBackward0>)
# tensor([[102400.],
#         [102400.],
#         [102400.]])
# tensor([[ True],
#         [ True],
#         [False]])

# Sometimes a.grad == d/a, sometimes not, when "a" is not a scalar.



