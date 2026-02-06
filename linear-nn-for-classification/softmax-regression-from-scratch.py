import torch
from d2l import torch as d2l

# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
#
# print(X.sum(0, keepdims=True))
# tensor([[5., 7., 9.]])
#
# print(X.sum(1, keepdims=True))
# tensor([[ 6.],
#         [15.]])

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here.

# X = torch.rand((2, 5))
# print(X)
# tensor([[0.4304, 0.5319, 0.1952, 0.6080, 0.9351],
#         [0.3760, 0.4580, 0.2769, 0.9808, 0.2990]])
#
# X_prob = softmax(X)
# print(X_prob)
# tensor([[0.1740, 0.1926, 0.1375, 0.2078, 0.2882],
#         [0.1740, 0.1888, 0.1576, 0.3185, 0.1611]])
#
# print(X_prob.sum(1))
# tensor([1.0000, 1.0000])


"""
Recall that cross-entropy takes the negative log-likelihood of the predicted probability assigned to the true label.
For efficiency we avoid Python for-loops and use indexing instead. In particular, the one-hot encoding in y_vec allows
us to select the matching terms in y_hat_vec.

To see this in action we create sample data y_hat with 2 examples of predicted probabilities over 3 classes and their
corresponding labels y. The correct labels are 0 and 2 respectively (i.e., the first and third class).
Using y as the indices of the probabilities in y_hat, we can pick out terms efficiently.
"""
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([
#     [0.1, 0.3, 0.6],
#     [0.3, 0.2, 0.5]
# ])
# print(len(y_hat))  # 2
# print(range(len(y_hat)))  # range(0, 2)
# print(list(range(len(y_hat))))  # [0, 1]
#
# print(y_hat[[0, 1], y])
# tensor([0.1000, 0.5000])


def cross_entropy(y_hat, y):
    # The first index is the list of row indices in y_hat.
    # The second index is the column indices defined by y.
    # The result (within the log) is a row of the y_hat probabilities corresponding to the correct labels in y.
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()


class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return softmax(torch.matmul(X, self.W) + self.b)

    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)


data = d2l.FashionMNIST(batch_size=256)
# 784 = 28 * 28, the number of pixels in each image.
# 10 = the number of classes in the dataset.
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

d2l.plt.show()

X, y = next(iter(data.val_dataloader()))
# The predictions will be the index of the largest element in the output vector,
# which corresponds to the predicted class.
preds = model(X).argmax(axis=1)
print(preds.shape)

wrong = preds.type(y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)

d2l.plt.show()

print(wrong.shape)