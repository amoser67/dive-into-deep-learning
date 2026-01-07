import math
import numpy as np
from d2l import torch as d2l
import torch

def f(x):
    return np.abs(x)

x = np.arange(-5, 5, .2)

d2l.plot(
    x,
    [f(x)],
    'x',
    'f(x)',
    legend=['f(x)']
)
d2l.plt.show()


"""
4. Let f(x) = sin(x). Plot the graph of f and its derivative f' using automatic differentiation.

    def f(x):
        return torch.sin(x)
    
    x0 = torch.arange(-5., 5., 0.2)
    
    # Graph of f.
    d2l.plot(
        x0, [f(x0)],
        "x", "f(x)",
        legend=["f(x)"]
    )
    d2l.plt.show()
    
    
    def df_dx(x1):
        x1.requires_grad_(True)
        y = f(x1)
        y.backward(gradient=torch.ones(len(y)))
        return x1.grad
    
    
    # Graph of f'.
    d2l.plot(
        x0, [df_dx(x0.clone())],
        "x", "f'(x)",
        legend=["f'(x)"]
    )
    d2l.plt.show()
"""

"""
Forward vs Backward Propagation

(From comment)

I tried doing forwardprop by hand on a number of computational graphs. For each input, one has to perform a
full forward pass through the graph. In contexts where we’d like to track the gradient of some output with
respect to one or a few specific inputs, forwardprop makes sense! It would also make sense in any context
where we have N inputs and M outputs, and N << M. In practice, neural networks have a massive number of
inputs and a scalar output loss, making backwards differentiation the obvious choice. A rule of thumb is
that the forward-mode cost is proportional to the number of inputs, and the backward-mode cost is proportional
to the number of outputs.

Other tradeoffs may come up in practice, depending on the layout of your computational graph. For example,
backprop starts off at a scalar, and the computational graph fans out from there. You might think this makes it
difficult to parallelize early parts of the graph, but in practice, backprop parallelizes very well. One might think
that forwardprop would be easier to parallelize (since you can start out propogating from all inputs in parallel), but
you could get weird dependency bottlenecks as different branches have to be “merged” deeper into the network: if
branch A from input a and branch B from input b must be multiplied together, and branch A is very fast to propogate
through but branch B is very slow to propogate through, branch A will be bottlenecked by branch B. Generally, you’ll
be bottlenecked by the slowest path through the network.

Another tradeoff is memory usage: backprop requires storing intermediate activations from the forward pass, meaning
memory usage scales with the depth of a network. Forwardprop doesn’t have this requirement! Aside from tracking the
partial derivatives at each step, forwardprop stateless as we move through the network, meaning the memory footprint
is generally much smaller.
"""







