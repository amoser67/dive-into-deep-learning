import torch

"""

Preliminaries > 2.1 Data Manipulation

"""

"""
2.1.1 Getting Started
"""
x = torch.arange(12, dtype=torch.float32)
print(x)

num_elements_in_x = x.numel()
print(num_elements_in_x)

# Transform x into a matrix with 3 rows and 4 columns.
# x[3] == X[0, 3], where 0 is the row index and 3 is the column index.
X = x.reshape(3, 4)
print(X)
# X = tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])
# Instead of explicitly stating both 3 and 4, we could've just stated one and set the other to -1, which tells
# the program to infer the -1 component automatically. E.g. x.reshape(3, 4) == x.reshape(3, -1) == x.reshape(-1, 4).

# Create a tensor whose values are all 0.
zeros = torch.zeros((2, 3, 4))
# (2, 3, 4) shape => an array of two matrices, each with 3 rows and 4 columns.

# Create a tensor whose values are all 1.
ones = torch.ones((2, 3, 4))

# Create a tensor with elements drawn from a normal distribution with mean 0 and std dev 1.
normal_dist = torch.randn(3, 4)

# Create a tensor with explicitly defined values.
explicitly_defined = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


"""
2.1.2 Indexing and Slicing
"""

# When only one index is specified for a kth order tensor, it is applied along axis 0.
print(X[-1])
# tensor([ 8.,  9., 10., 11.])
print(X[1:3])
# tensor([[ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

# We can also write elements of a matrix by specifying indices.
X[1, 2] = 17
print(X)
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5., 17.,  7.],
#         [ 8.,  9., 10., 11.]])

# If we want to assign multiple elements the same value, we apply the indexing on
# the left-hand side of the assignment operation. For instance, [:2, :] accesses
# the first and second rows, where : takes all the elements along axis 1 (column).
X[:2, :] = 12
print(X)
# tensor([[12., 12., 12., 12.],
#         [12., 12., 12., 12.],
#         [ 8.,  9., 10., 11.]])


"""
2.1.3 Operations
"""

# elementwise operations apply a scalar operation to each element of a tensor.
# For functions that take two tensors as inputs, elementwise operations apply
# some standard binary operator on each pair of corresponding elements.

# unary scalar operators (taking one input) are denoted by the signature f: R -> R.
print(torch.exp(x))
# tensor([162754.7969, 162754.7969, 162754.7969, 162754.7969, 162754.7969,
#         162754.7969, 162754.7969, 162754.7969,   2980.9580,   8103.0840,
#          22026.4648,  59874.1406])


# binary scalar operators (taking two inputs) are denoted by the signature f: R,R -> R
a = torch.tensor([1.0, 2, 4, 8])
b = torch.tensor([2, 2, 2, 2])
print(a + b,   # tensor([ 3.,  4.,  6., 10.])
      a - b,   # tensor([-1.,  0.,  2.,  6.])
      a * b,   # tensor([ 2.,  4.,  8., 16.])
      a / b,   # tensor([0.5000, 1.0000, 2.0000, 4.0000])
      a ** b)  # tensor([ 1.,  4., 16., 64.])

# We can concatenate tensors by providing a list of tensors and specifying along which axis to concatenate.
A = torch.arange(12, dtype=torch.float32).reshape((3, 4))
B = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((A, B), dim=0))
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]])
print(torch.cat((A, B), dim=1))
# tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])

# Constructing a binary tensor via logical statements.
print(A == B)
# tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]])

# Sum all elements of a tensor
print(A.sum())
# tensor(66.)


"""
2.1.4 Broadcasting

Under certain conditions, even when the shapes of two tensors differ, we can still perform
elementwise binary operations by invoking the broadcasting mechanism.

Broadcasting works according to the following two-step procedure:
    (i)  expand one or both arrays by copying elements along axes with length 1 so that after this transformation,
         the two tensors have the same shape;
         
    (ii) perform an elementwise operation on the resulting arrays.
"""
c = torch.arange(3).reshape((3, 1))  # 3 x 1 matrix
d = torch.arange(2).reshape((1, 2))  # 1 x 2 matrix
print(c, d)
# tensor([[0],
#         [1],
#         [2]])
# tensor([[0, 1]])

"""
c = tensor([[0],
            [1],
            [2]])
d = tensor([[0, 1]])

i:
    c = [[0, 0],
         [1, 1],
         [2, 2]]
         
     d = [[0, 1],
          [0, 1],
          [0, 1]]
          
    c + d = [[0, 1],
             [1, 2],
             [2, 3]]
"""
print(c + d)  # Creates a 3 x 2 matrix.
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])


"""
2.1.5 Saving Memory
"""

# Running B = A + B will dereference the tensor that B used to point to.
# We can check this by using Python's id() function, which returns the address
# of the referenced object in memory.
before = id(B)
B = B + A
print(id(B) == before)  # False

# In general, we often want to update values in place, to reduce the number of memory allocations.
# This can be done using the slice notation: Y[:] = <expression>.
C = torch.zeros_like(B)
print(id(C))  # 2051333632832
C[:] = A + B
print(id(C))  # 2051333632832

# This can be done with an element from <expression> as well.
# A[:] = A + B (this can also be written as A += B)


"""
2.1.6 Conversion to Other Python Objects
"""

D = A.numpy()
F = torch.from_numpy(D)
print(type(D), type(F))
# <class 'numpy.ndarray'> <class 'torch.Tensor'>

# To convert a size-1 tensor to a Python scalar, we can invoke the item function or Python’s built-in functions.
d = torch.tensor([3.5])
print(
    d,          # tensor([3.5000])
    d.item(),   # 3.5
    float(d),   # 3.5
    int(d))     # 3

"""
2.1 Exercises

1. Run the code in this section. Change the conditional statement X == Y to X < Y or X > Y,
and see what kind of tensor you get.

2. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes,
e.g. 3-dim tensors. Is the result the same as expected?
"""

"""
2.

notes:

m = [
"""

# j = torch.arange(6).reshape((3, 2))
# k = torch.arange(6).reshape((2, 3))
# print(j + k)
# The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1

# j = torch.arange(12).reshape((3, 2, 2))
# k = torch.arange(12).reshape((2, 2, 3))
# print(j + k)
# RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2.

j = torch.arange(6).reshape((3, 2, 1))
k = torch.arange(6).reshape((1, 2, 3))
print(j)
print(k)
print(j.shape)
print(k.shape)
print(j + k)
# tensor([[[ 0,  1,  2],
#          [ 4,  5,  6]],
#
#         [[ 2,  3,  4],
#          [ 6,  7,  8]],
#
#         [[ 4,  5,  6],
#          [ 8,  9, 10]]])

"""
3 x 2 x 1
3, 2 x 1 matrices
j = tensor([
            [
                [0],
                [1]
            ],
    
            [
                [2],
                [3]
            ],
    
            [
                [4],
                [5]
            ]
           ])

1 x 2 x 3
1, 2 x 3 matrix
k = tensor([[[0, 1, 2],
             [3, 4, 5]]])
             
So, k needs to 2 more matrices, and j needs to make each matrix 2 x 3.

j = [
        [
            [0, 0, 0],
            [1, 1, 1]
        ],
        
        [
            [2, 2, 2],
            [3, 3, 3]
        ],
        [
            [4, 4, 4],
            [5, 5, 5]
        ]
    ]
    
k = [
        [
            [0, 1, 2],
            [3, 4, 5]
        ],
        [
            [0, 1, 2],
            [3, 4, 5]
        ],
        [
            [0, 1, 2],
            [3, 4, 5]
        ],
    ]   

j + k = [
            [
                [0, 1, 2],
                [4, 5, 6],
            ],
            ...
        ]   
        

3 x 2 x 3 Tensor:
tensor([
            [
                [ 0,  1,  2],
                [ 4,  5,  6]
            ],

            [
                [ 2,  3,  4],
                [ 6,  7,  8]
            ],

            [
                [ 4,  5,  6],
                [ 8,  9, 10]
            ]
        ])
        
3, 2 x 3 matrices


"""
