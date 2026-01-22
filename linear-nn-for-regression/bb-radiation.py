import math
import torch
from d2l import torch as d2l

# Constants
c = 299_792_458  # m/s
h = 6.62607015e-34  # J/Hz
k = 1.380649e-23  # J/K

# Black body radiation.
def B(wavelength, temp):
    return (2 * h * c**2 / wavelength**5) * (1 / torch.exp(h*c / (wavelength*k*temp)))


# Y is ln(B * wavelength**5)
# def compute_y(wavelength, temp):
#     return torch.log( (-h * c / (k * temp)) * (1 / wavelength) + math.log(2 * h * c) )


def wavelength_to_x(wavelength):
    return 1 / wavelength


def temp_to_w(temp):
    return (-h * c / k) * (1 / temp)


def energy_to_y(energy, wavelength):
    return torch.log(energy * wavelength**5)


"""
Create training data.
    
    B is y
    wavelength is x
    temperature is y
    
    desired w: 0.00028775537550078677
"""

seed_wavelength_sorted = torch.arange(.1, 100, .1, dtype=torch.float)
# seed_temp_sorted = torch.arange(-100, -.1, .1, dtype=torch.float)

seed_wavelength = seed_wavelength_sorted[torch.randperm(seed_wavelength_sorted.nelement())]
seed_temp = 50.00
seed_energy = B(seed_wavelength, seed_temp)

seed_x = wavelength_to_x(seed_wavelength)
seed_w = temp_to_w(seed_temp)

# seed_x should be wavelength, not 1/wavelength.
seed_y = energy_to_y(seed_energy, seed_wavelength)
seed_b = math.log(2 * h * c**2)
seed_y[5] = 10000
# torch.set_printoptions(precision=10)
# print(seed_y)
# print('{0:.16f}'.format(seed_y[4]))
#
# print("X")
# print(seed_x[0:10])
# print("Energy")
# print(seed_energy[0:10])
# print("Wavelength")
# print(seed_wavelength[0:10])
print("W")
print(seed_w)
#
# print("Y")
# print(seed_y)
#
print("B")
print(seed_b)


# print(seed_x.shape)
#
# print(seed_b)
# print(torch.tensor([seed_w]))


# print("X")
# print(seed_x)
# print("W")
# print(seed_w)
# print("Y")
# print(seed_y)
# #


# seed_y = compute_y(seed_x, seed_w)


class LinearRegressionScratch(d2l.Module):
    # lr = learning rate
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        # w is a column vector with num_inputs rows.
        # The values of w are selected from a normal distribution with mean 0 and std .01 (variance = .0001).
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        print("self w", self.w)

        # A 1x1 tensor
        self.b = torch.zeros(1, requires_grad=True)
        print("self b", self.b)

    def forward(self, X):
        # returns Xw + b
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        # Returns the average of the squared losses.
        # Squared losses are divided by 2 to simplify derivation.
        # l = (y_hat - y) ** 2 / 2
        l = (y_hat - y) ** 2 / 2
        return l.mean()
        # l = (y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()
        # return l

    def configure_optimizers(self):
        # Returns an instance of the SGD class, initialized with the relevant model properties.
        return SGD([self.w, self.b], self.lr)


class SGD(d2l.HyperParameters):
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        # Subtract the param's gradient (multiplied by the learning rate) from each param.
        # Need to better understand the process of subtracting the gradient.
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        # Sets all gradients to 0. Necessary to run before a backpropagation step.
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class SeededRegressionData(d2l.DataModule):
    """Synthetic data for linear regression.

    Defined in :numref:`sec_synthetic-regression-data`"""
    def __init__(self, w, b, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.X = torch.reshape(seed_x, (num_train + num_val, 1))
        self.y = seed_y

    def get_dataloader(self, train):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch


@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1


model = LinearRegressionScratch(1, 0.01)
data = SeededRegressionData(torch.tensor([seed_w]), seed_b, num_train=500, num_val=499)
trainer = d2l.Trainer(max_epochs=50)
trainer.fit(model, data)

print("Model W")
print(model.w)

with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')

d2l.plt.show()

print("Model B")
print(model.b)

"""
Mean squared error results in closer estimates when outliers aren't present.
Absolute value loss handles outliers better, since the differences aren't squared.

How can we combine both?

Maybe have a condition that uses one alg if difference is less than X and another if it isn't.
"""
