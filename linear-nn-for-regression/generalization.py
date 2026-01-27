"""
Exercises
"""

"""
1. When can you solve the problem of polynomial regression exactly?

    When each training example is distinct and the polynomial has degree equal to the number of training examples.


2. Give at least five examples where dependent random variables make treating the problem as IID data inadvisable.
    
        - Predicting temperature in a city by training the algorithm on data from the whole country.
        - Predicting the gas mileage of a car by taking the gas mileage of all cars.
        - Predicting the highway gas mileage of a car by taking the gas mileage of the car on different types of roads.
        - Predicting the temperature at 5:00pm by taking historical temperature data from throughout the days (past and present).
        - Predicting political beliefs of a US citizen by training the algorithm on Texans.
           

3. Can you ever expect to see zero training error? Under which circumstances would you see zero generalization error

    In special cases like #1, or when you have an extremely/overly expressive model, you may be able to perfectly fit
    the training data, though it's unlikely in real world datasets/applications.
    
    Outside of trivial problems or infinite training data, it seems unlikely we would ever see zero generalization error.
    
    
4. Why is K-fold cross-validation very expensive to compute?
    
    Because if you have k training examples, you must train k sets of k-1 examples. This means that much of the data
    overlaps, making the per example training cost quite high.
    
    
5. Why is the K-fold cross-validation error estimate biased?
    
    The training error for a model is generally biased, and that bias gets compounded when we maximize the re-use of
    training data, especially given that we expect the amount of training data to be limited given that k-fold
    cross-validation was chosen.
"""