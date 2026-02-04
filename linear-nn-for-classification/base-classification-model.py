import torch
from d2l import torch as d2l


class Classifier(d2l.Module):
    """The base class for classification models."""

    def validation_step(self, batch):

# Process the batch.

        Y_hat = self(*batch[:-1])

# Plot the loss and classification accuracy.

        self.plot("loss", self.loss(Y_hat, batch[-1]), train=False)
        self.plot("acc", self.accuracy(Y_hat, batch[-1]), train=False)

# We use a SGD optimizer, operating on minibatches, just as we did for lin regression.

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        # Y_hat is a (we assume) a matrix at this point anyway, unclear why this is necessary.
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        predictions = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (predictions == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare