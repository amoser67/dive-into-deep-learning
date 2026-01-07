import torch

"""

2.3 Linear Algebra

"""


"""
2.3.1 Scalars
"""
# Scalars can be represented as tensors with only one element.
scalar_x = torch.tensor(3.0)
scalar_y = torch.tensor(2.0)
# print(
#     scalar_x + scalar_y,  # tensor(5.)
#     scalar_x * scalar_y,  # tensor(6.)
#     scalar_x / scalar_y,  # tensor(1.5000)
#     scalar_x ** scalar_y  # tensor(9.)
# )


"""
2.3.2 Vectors

For current purposes, vectors can be thought of as fixed length arrays of scalars.

Vectors are implemented as 1st order tensors.

Linear algebra subscripts start at 1 (not 0).

By default, we visualize vectors as a column.
"""

vector_x = torch.arange(3)
# print(vector_x)
# tensor([0, 1, 2])
# print(vector_x[2])  # tensor(2)

# The dimensionality of a vector is equivalent to the length of the tensor (i.e. len(tensor_x))

# Sometimes, people refer to the number of axes in a vector as dimensionality as well,
# so for clarity, we use "dimensionality" to exclusively refer to the number of elements in a vector
# and "order" to refer to the number of axes.


"""
2.3.3 Matrices

Scalars are 0-order tensors.
Vectors are 1st-order tensors.
Matrices are 2nd-order tensors.

The shape of a matrix is (m, n), where m is the number of rows, and n is the number of columns.
"""

A = torch.arange(6).reshape(3, 2)

# print(A)
# tensor([[0, 1],
#         [2, 3],
#         [4, 5]])

# Transpose A:
# A_transpose = A.T
# print(A_transpose)
# tensor([[0, 2, 4],
#         [1, 3, 5]])

# Symmetric matrices are those that are equal to their own transpose.
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(B == B.T)
# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True]])


"""
2.3.4 Tensors
"""

C = torch.arange(24).reshape(2, 3, 4)
print(C)
print(C.sum(axis=1).shape)
print(C.sum(axis=1))
print(C.sum(axis=2).shape)
# print(C)
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])


"""
2.3.5 Basic Properties of Tensor Arithmetic
"""

# We can clone a tensor to copy it by value.
D = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print(D.sum(axis=0))
E = D.clone()
# We can perform elementwise operations on tensors.
# print(D)
# tensor([[0., 1., 2.],
#         [3., 4., 5.]])
# print(D + E)
# tensor([[ 0.,  2.,  4.],
#         [ 6.,  8., 10.]])


"""
2.3.6 Reduction
"""
# We can calculate the sum of all of a tensors elements using the sum method.
# This method "reduces" a tensor to a scalar.
F = torch.arange(3, dtype=torch.float32)
# print(F, F.sum()) #  tensor([0., 1., 2.]) tensor(3.)

# print(D.shape, D.sum())  # torch.Size([2, 3]) tensor(15.)

# We can also use sum to reduce one of the tensor's axes to a scalar.
# print(D.sum(axis=0).shape)  # torch.Size([2, 3]) tensor(15.)
# print(D.sum(axis=[0, 1]) == D.sum())  # tensor(True)

# We can calculate the mean of a tensor's (or one of its axes') values by using the
# "mean" method or via the traditional approach.
# print(D.mean(axis=0) == D.sum(axis=0) / D.shape[0])  # tensor([True, True, True])


"""
2.3.7 Non-Reduction Sum

Sometimes it can be useful to keep the number of axes unchanged when invoking the function
for calculating the sum or mean. This matters when we want to use the broadcast mechanism.
"""

sum_D = D.sum(axis=1, keepdims=True)
# print(sum_D, sum_D.shape)
# tensor([[ 3.],
#         [12.]])
# torch.Size([2, 1])

# As opposed to:
sum_D_no_keepdims = D.sum(axis=1)
# print(sum_D_no_keepdims, sum_D_no_keepdims.shape)
# tensor([ 3., 12.])
# torch.Size([2])

# For example, since sum_D keeps its two axes after summing each row, we can divide A by
# sum_A with broadcasting to create a matrix where each row sums up to 1.
# print(D / sum_D)
# tensor([[0.0000, 0.3333, 0.6667],
#         [0.2500, 0.3333, 0.4167]])

# We can calculate the cumulative sum of elements of D along some axis, say axis=0 (row by row),
# using the cumsum function. By design, this function does not reduce the input tensor along any axis.
# print(D.cumsum(axis=0))
# tensor([[0., 1., 2.],
#         [3., 5., 7.]])
# Basically, the values in each row are the sum of the original values and all values
# in previous rows of the corresponding column.

"""
2.3.8 Dot Products

The dot product of two vectors is the sum over the products of the elements at the same position.
"""
a = torch.ones(3, dtype=torch.float32)
b = torch.arange(3, dtype=torch.float32)

# print(a)  # tensor([1., 1., 1.])
# print(b)  # tensor([0., 1., 2.])
# print(torch.dot(a, b))  # tensor(3.)

# Equivalently, we can calculate the dot product by summing the results of elementwise multiplication.
# print(torch.sum(a * b))  # tensor(3.)


"""
2.3.9 Matrix-Vector Products
"""
# print(D.shape)  # torch.Size([2, 3])
# print(a.shape)  # torch.Size([3])
# print(torch.mv(D, a))  # tensor([ 3., 12.])
# # The @ operator executes matrix-matrix or matrix-vector products, depending on its arguments.
# print(D@a)  # tensor([ 3., 12.])


"""
2.3.10 Matrix-Matrix Multiplication
"""

G = torch.ones(3, 4)
# print(D)
# tensor([[0., 1., 2.],
#         [3., 4., 5.]])
# print(G)
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
# print(torch.mm(D, G))
# tensor([[ 3.,  3.,  3.,  3.],
#         [12., 12., 12., 12.]])
# print(D@G)
# tensor([[ 3.,  3.,  3.,  3.],
#         [12., 12., 12., 12.]])


"""
2.3.11 Norms
"""

# l2 norm
u = torch.tensor([3.0, -4.0])
# print(torch.norm(u))
# tensor(5.)

# l1 norm
# print(torch.abs(u).sum())
# tensor(7.)

# Frobenius norm
# print(torch.norm(torch.ones((4, 9))))
# tensor(6.)

