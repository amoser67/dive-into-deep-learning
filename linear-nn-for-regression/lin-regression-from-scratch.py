import torch
from d2l import torch as d2l


class LinearRegressionScratch(d2l.Module):
    # lr = learning rate
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        # w is a column vector with num_inputs rows.
        # The values of w are selected from a normal distribution with mean 0 and std .01 (variance = .0001).
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)

        # A 1x1 tensor
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        # returns Xw + b
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        # Returns the average of the squared losses.
        # Squared losses are divided by 2 to simplify derivation.
        l = (y_hat - y) ** 2 / 2
        return l.mean()

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


"""
Training

In each epoch:
    We iterate through the entire training dataset, passing once through every example
    (assuming that the number of examples is divisible by the batch size).
    
In each iteration:
    we grab a minibatch of training examples, and compute its loss through the model’s training_step method.
    
Then we compute the gradients with respect to each parameter.

Finally, we will call the optimization algorithm to update the model parameters.

In summary, we will execute the following loop:

    - Initialize parameters (w, b)
    
    - Repeat until done:
        - Compute gradient g by taking partial derivative of loss function.
        - Update parameters (w, b) <- (w, b) - ηg        
        

"""

@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
    return batch


@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train()

    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # to be discussed later
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


model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)

with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')

print(data)
# error in estimating w: tensor([ 0.1047, -0.1969])
# error in estimating b: tensor([0.2140])

