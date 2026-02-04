import torch
from torch import nn
from d2l import torch as d2l


class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD(
            [
                {'params': self.net.weight, 'weight_decay': self.wd},
                {'params': self.net.bias}
            ],
            lr=self.lr
        )

class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)


def l2_penalty(w):
    return (w ** 2).sum() / 2


def l1_penalty(w):
    return torch.sum(torch.abs(w))


"""
Experiment with the value of lambda in the estimation problem in this section.
Plot training and validation accuracy as a function of lambda. 
What do you observe?

    As lambda increased from 0 to 20, validation loss decreased dramatically.
    
"""

lambdas = [0, 1]
# lambdas = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20]
losses = []
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
for lambd in lambdas:
    trainer = d2l.Trainer(max_epochs=10)
    model = WeightDecay(wd=lambd, lr=0.01)
    trainer.fit(model, data)
    d2l.plt.show()
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
    # w, b = model.get_w_b()
    # losses.append((data.w - w.reshape(data.w.shape)).sum())






