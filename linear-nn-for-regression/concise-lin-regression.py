import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        # Mean squared error loss without the 1/2 factor.
        fn = nn.MSELoss()
        # fn = nn.HuberLoss()
        # fn = nn.SmoothL1Loss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        print(self.net.weight.grad)
        return (self.net.weight.data, self.net.bias.data)


# model = LinearRegression(lr=0.01)
# data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# trainer = d2l.Trainer(max_epochs=12)
# trainer.fit(model, data)
#
# w, b = model.get_w_b()
#
# print(f"Error in estimating w: {data.w - w.reshape(data.w.shape)}")
# print(f"Error in estimating b: {data.b - b}")

"""
Exercises

1. How would you need to change the learning rate if you replace the
"aggregate loss over the minibatch" with an "average over the loss on the minibatch".

    If aggregate means just summing the squared differences, then we would likely want
    the learning rate to be lower, since the loss values would be higher.


4. What is the effect on the solution if you change the learning rate and the number of epochs?
Does it keep on improving?

Base result (lr=0.03, max_epochs=3)
Error in estimating w: tensor([ 0.0086, -0.0097])
Error in estimating b: tensor([0.0162])

Error in estimating w: tensor([ 0.0034, -0.0135])
Error in estimating b: tensor([0.0080])


Results for lower learning rate  (0.01)
Error in estimating w: tensor([ 0.2654, -0.5852])
Error in estimating b: tensor([0.4978])

Error in estimating w: tensor([ 0.2379, -0.5230])
Error in estimating b: tensor([0.5981])

Results for lower learning rate (0.01) and more epochs (9)

Error in estimating w: tensor([ 0.0056, -0.0091])
Error in estimating b: tensor([0.0082])

Error in estimating w: tensor([ 0.0032, -0.0048])
Error in estimating b: tensor([0.0130]

Results for lower learning rate (0.01) and more epochs (12)

Error in estimating w: tensor([ 0.0006, -0.0011])
Error in estimating b: tensor([0.0021])

Error in estimating w: tensor([ 3.6240e-05, -2.2647e-03])
Error in estimating b: tensor([0.0018])


Conclusion:
Reducing the lr and increasing the number of epochs provides better results. However, we expect this will
have diminishing returns, and eventually result in overfitting, as the number of epochs increases. Though,
I suppose if the learning rate continues to decrease in proportion with increase in epochs, you may be able
to avoid overfitting. However, you would still get diminishing returns and increased execution time.

"""

"""
5. Plot the estimation error for w_hat - w and b_hat - b as a function of the amount of data.
Hint: increase the amount of data logarithmically rather than linearly, i.e., 5, 10, 20, 50, …, 10,000
rather than 1000, 2000, …, 10,000.

    We get diminishing returns pretty quickly.
    
Why is the hint appropriate?

    We want the amount of data to increase as a ratio of the previous values, since we need large
    relative increases to identify trends despite the diminishing returns.
"""

data_size_arr = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10_000]
w_error_arr = []
b_error_arr = []

for data_size in data_size_arr:
    model = LinearRegression(lr=0.03)
    data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2, num_train=data_size, num_val=data_size)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)
    w, b = model.get_w_b()
    w_error_arr.append((data.w - w.reshape(data.w.shape)).sum())
    b_error_arr.append(data.b - b)


d2l.plot(
    data_size_arr,
    [w_error_arr, b_error_arr],
    "num data points",
    "W error",
    legend=["w error", "b error"],
    figsize=(8, 6),
)
d2l.plt.show()
