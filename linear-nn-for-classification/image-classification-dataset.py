import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()


class FashionMNIST(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()

        self.save_hyperparameters()

# A function/transform that takes in a PIL image and returns a transformed version.

        trans = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

# Training dataset

        self.train = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=True,
            transform=trans,
            download=True
        )

# Validation dataset

        self.val = torchvision.datasets.FashionMNIST(
            root=self.root,
            train=False,
            transform=trans,
            download=True
        )


    @staticmethod
    def text_labels(indices):
        """Return text labels."""
        labels = ["t-shirt", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle boot"]
        return [labels[i] for i in indices]


    def get_dataloader(self, train):
        dataset = self.train if train else self.val

        return torch.utils.data.DataLoader(
            dataset,  # dataset from which to load the data.
            self.batch_size,
            shuffle=train,  # Shuffles the data at every epoch if True.
            num_workers=self.num_workers  # How many subprocesses to use for data loading (default: 0).
        )

    def visualize(self, batch, num_rows=1, num_cols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        # 'Squeeze': Returns a tensor with all specified dimensions of input of size 1 removed.
        # The returned tensor shares the storage with the input tensor, so changing the contents of
        # one affects the other.
        d2l.show_images(X.squeeze(1), num_rows, num_cols, titles=labels)


# View the shape of the data.
# data = FashionMNIST(resize=(32, 32))
# print(len(data.train), len(data.val))
# print(data.train[0][0].shape)

# Measure the time it takes to read the images.
# X, y = next(iter(data.train_dataloader()))  # train_dataloader() => get_dataloader(train=True)
# tic = time.time()
# for X, y in data.train_dataloader():
#     continue
# toc = time.time()
# print(f'{toc - tic:.2f} sec')

# Visualize a batch of images.
# batch = next(iter(data.val_dataloader()))
# data.visualize(batch)
# d2l.plt.show()

"""
Notes

- Fashion-MNIST is an apparel classification dataset consisting of images representing 10 categories. 
- Images are read as tensors of shape (batch_size, num_channels, height, width).
- Data iterators are a key component for efficient performance.
"""

"""
Exercises
"""
# Q1. Does reducing the batch size (for instance, to 1) affect the reading performance?
#
# data = FashionMNIST(batch_size=1, resize=(32, 32))
# X, y = next(iter(data.train_dataloader()))  # train_dataloader() => get_dataloader(train=True)
# tic = time.time()
# for X, y in data.train_dataloader():
#     continue
# toc = time.time()
# print(f'{toc - tic:.2f} sec')  # 73.12 sec
#
# A1. Yes, reducing the batch size to 1 significantly increases the time taken to read the images.

"""
Q2. Is the current data loader implementation fast enough?
    Explore ways to improve it.
    Use a system profiler to find out where the bottlenecks are.
    
       1    0.000    0.000    0.001    0.001 dataloader.py:1(<module>)
       10    0.000    0.000    0.000    0.000 dataloader.py:1082(<genexpr>)
        2    0.000    0.000    0.026    0.013 dataloader.py:1087(_reset)
        2    0.000    0.000    0.000    0.000 dataloader.py:1103(<listcomp>)
      939    0.002    0.000    3.168    0.003 dataloader.py:1120(_try_get_data)
        1    0.000    0.000    0.000    0.000 dataloader.py:124(DataLoader)
      939    0.002    0.000    3.170    0.003 dataloader.py:1266(_get_data)
      940    0.007    0.000    3.242    0.003 dataloader.py:1299(_next_data)
      955    0.005    0.000    0.083    0.000 dataloader.py:1348(_try_put_index)
      939    0.001    0.000    0.059    0.000 dataloader.py:1368(_process_data)
        8    0.000    0.000    0.000    0.000 dataloader.py:1375(_mark_worker_as_unavailable)
        3    0.000    0.000    0.021    0.007 dataloader.py:1401(_shutdown_workers)
        2    0.000    0.000    0.016    0.008 dataloader.py:1478(__del__)
        2    0.000    0.000    0.000    0.000 dataloader.py:226(__init__)
        2    0.000    0.000    0.064    0.032 dataloader.py:382(_get_iterator)
        2    0.000    0.000    0.000    0.000 dataloader.py:389(multiprocessing_context)
        2    0.000    0.000    0.000    0.000 dataloader.py:393(multiprocessing_context)
    40/38    0.000    0.000    0.000    0.000 dataloader.py:417(__setattr__)
        2    0.000    0.000    0.064    0.032 dataloader.py:426(__iter__)
        6    0.000    0.000    0.000    0.000 dataloader.py:441(_auto_collation)
        2    0.000    0.000    0.000    0.000 dataloader.py:445(_index_sampler)
        4    0.000    0.000    0.000    0.000 dataloader.py:486(check_worker_number_rationality)
        1    0.000    0.000    0.000    0.000 dataloader.py:564(_BaseDataLoaderIter)
        2    0.000    0.000    0.001    0.000 dataloader.py:565(__init__)
        2    0.000    0.000    0.000    0.000 dataloader.py:610(_reset)
      955    0.001    0.000    0.037    0.000 dataloader.py:620(_next_index)
      940    0.008    0.000    3.325    0.004 dataloader.py:626(__next__)
        1    0.000    0.000    0.000    0.000 dataloader.py:658(_SingleProcessDataLoaderIter)
        1    0.000    0.000    0.000    0.000 dataloader.py:681(_MultiProcessingDataLoaderIter)
        1    0.000    0.000    0.000    0.000 dataloader.py:70(_DatasetKind)
        1    0.000    0.000    0.000    0.000 dataloader.py:82(_InfiniteConstantSampler)
        2    0.000    0.000    0.000    0.000 dataloader.py:93(_get_distributed_settings)
        2    0.001    0.000    0.064    0.032 dataloader.py:991(__init__)
        
        
    A2. Could increase the batch size and perhaps decouple the data loading from the training loop.
"""

data = FashionMNIST(resize=(32, 32))

X, y = next(iter(data.train_dataloader()))  # train_dataloader() => get_dataloader(train=True)
tic = time.time()
for X, y in data.train_dataloader():
    continue
toc = time.time()
print(f'{toc - tic:.2f} sec')
