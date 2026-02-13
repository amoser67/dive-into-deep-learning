import torch
from torch import nn
from d2l import torch as d2l


# class MLPScratch(d2l.Classifier):
#     def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
#         super().__init__()
#         self.save_hyperparameters()
#         # nn.Parameter automatically registers a class attribute as a parameter to be tracked by autograd.
#         self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
#         self.b1 = nn.Parameter(torch.zeros(num_hiddens))
#         self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
#         self.b2 = nn.Parameter(torch.zeros(num_outputs))
#
#     def forward(self, X):
#         X = X.reshape((-1, self.num_inputs))
#         H = relu(torch.matmul(X, self.W1) + self.b1)
#         return torch.matmul(H, self.W2) + self.b2
#
#
# def relu(X):
#     # A tensor filled with 0s with the same shape and type as X.
#     a = torch.zeros_like(X)
#     return torch.max(X, a)
#
#
# model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
# data = d2l.FashionMNIST(batch_size=256)
# trainer = d2l.Trainer(max_epochs=10)
# trainer.fit(model, data)
# d2l.plt.show()

class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens),
            nn.ReLU(),
            nn.LazyLinear(num_outputs)
        )


# model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
# trainer = d2l.Trainer(max_epochs=8)
# trainer.fit(model, data)
# d2l.plt.show()

learning_rates = [0.05, 0.1, 0.3]
for lr in learning_rates:
    model = MLP(num_outputs=10, num_hiddens=256, lr=lr)
    trainer = d2l.Trainer(max_epochs=8)
    trainer.fit(model, data)
    d2l.plt.show()


"""
Exercises:
1. Change the number of hidden units num_hiddens and plot how its number affects the accuracy of the model.
What is the best value of this hyperparameter?

    Sweet spot seems to be around 256. Higher and the model tends to overfit. Lower and loss is a bit higher.
    
2. Try adding a hidden layer to see how it affects the results.

    It seemed to peak a bit earlier and then start overfitting a bit. The values seemed pretty similar though.
    
4.  How does changing the learning rate alter your results?
    With all other parameters fixed, which learning rate gives you the best results?
    How does this relate to the number of epochs?
    
    0.1 Seemed to be a decent learning rate. 0.3 was a bit too high and the model seemed to diverge. 0.05 was a bit too
    low and the model seemed to converge a bit slower.
"""