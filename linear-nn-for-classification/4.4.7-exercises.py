import torch
from d2l import torch as d2l

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here.

X = torch.tensor([[100.0, 50.0, 25.0], [1.0, 2.0, 3.0]])
print(softmax(X))

X_2 = torch.tensor([
    [-101.0, -102.0, -103.0],
    [-104.0, -105.0, -106.0]
])

print(softmax(X_2))

"""
Question: Test whether softmax still works correctly if an input has a value of 100.
Answer: The result for that row is nan for the 100 input, and 0 for the rest.

    softmax([
        [100.0, 50.0, 25.0],
        [1.0,   2.0,  3.0]
    ])
    
    [
        [   nan, 0.0000, 0.0000],
        [0.0900, 0.2447, 0.6652]
    ]

Question: Test whether softmax still works correctly if the largest input is smaller than -100.
Answer: Doesn't work. Returns either all nan in a row, or inaccurate values:

    softmax([
        [-101.0, -102.0, -103.0],
        [-104.0, -105.0, -106.0]
    ])
    
    [
        [0.6667, 0.2667, 0.0667],
        [   nan,    nan,    nan]
    ]
            
Question: Implement a fix by looking at the value relative to the largest entry in the argument.
Answer: For large input:

    X = X - X.max(axis=1, keepdims=True)
        
"""
