import torch
from d2l import torch as d2l

# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(1, keepdims=True)
#     return X_exp / partition  # The broadcasting mechanism is applied here.
#
# X = torch.tensor([[100.0, 50.0, 25.0], [1.0, 2.0, 3.0]])
# print(softmax(X))
#
# X_2 = torch.tensor([
#     [-101.0, -102.0, -103.0],
#     [-104.0, -105.0, -106.0]
# ])
#
# print(softmax(X_2))

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

"""
Question: Implement a cross_entropy function that follows the definition of the cross-entropy loss function.

Answer:
    def cross_entropy_non_optimized(y_hat, y):
        return torch.dot(torch.log(y_hat), y)
    
Sub-question: Why is it slower?
Answer: Because we are using dot product which is wasteful since all but one of the components in y are zero.

Sub-question: When would it make sense to use it?
Answer: When y has more/all non-zero entries.

Sub-question: What do you need to be careful of?
Answer: y_hat must be greater than 0.
"""

"""
Question: Is it always a good idea to return the most likely label? For example, would you do this for a medical
diagnosis? How would you address this?

Answer: For medical diagnosis, it may be better to return the (non-negligible) probabilities of each diagnosis.
This is because there are often multiple explanations for symptoms, and each can have very different treatments,
in terms of cost, risk, etc., as well as different timelines.  
"""

"""
Question: Assume that we want to use softmax regression to predict the next word based on some features.
What are some problems that might arise from a large vocabulary?

Answer: The probabilities could get very small, leading to underflow issues. Holding everything else static and picking
the next word could be limiting, as opposed to picking/evaluating phrases or combinations of words.
"""