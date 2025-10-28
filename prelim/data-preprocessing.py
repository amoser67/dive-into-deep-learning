import os
import pandas as pd
import time
import torch

"""

Preliminaries > 2.2 Data Preprocessing

"""

"""
2.2.1 Reading the Dataset
"""

# Create CSV
os.makedirs(os.path.join("..", "data"), exist_ok=True)
data_file = os.path.join("..", "data", "house_tiny.csv")
with open(data_file, "w") as f:
    f.write("""NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000""")

# Read CSV
data = pd.read_csv(data_file)
# print(data)
#    NumRooms RoofType   Price
# 0       NaN      NaN  127500
# 1       2.0      NaN  106000
# 2       4.0    Slate  178100
# 3       NaN      NaN  140000


"""
2.2.2 Data Preparation
"""
# Note: Pandas converts NA or empty values to NaN.
# Missing values are a persistent menace in data science.

# We can separate out empty and non-empty columns
inputs = data.iloc[:, 0:2]  # iloc := integer-location based indexing
targets = data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)
#    NumRooms  RoofType_Slate  RoofType_nan
# 0       NaN           False          True
# 1       2.0           False          True
# 2       4.0            True         False
# 3       NaN           False          True

# For missing numerical values, a common heuristic is to replace the NaN entries with
# the mean value of the corresponding column.
inputs = inputs.fillna(inputs.mean())
# print(inputs)
#    NumRooms  RoofType_Slate  RoofType_nan
# 0       3.0           False          True
# 1       2.0           False          True
# 2       4.0            True         False
# 3       3.0           False          True


"""
2.2.3 Conversion to the Tensor Format
"""
# Now that all of our inputs and targets are numerical,
# we can convert them to the tensor format.
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
# print(X)
# tensor([[3., 0., 1.],
#         [2., 0., 1.],
#         [4., 1., 0.],
#         [3., 0., 1.]], dtype=torch.float64)
# print(y)
# tensor([127500., 106000., 178100., 140000.], dtype=torch.float64)


"""
Exercises
"""

"""
1.  Inspect some of the datasets from https://archive.ics.uci.edu/datasets.
"""

"""
2.  Try indexing and selecting data columns by name rather than by column number.

name_inputs = data.loc[:, ["RoofType", "NumRooms"]]
name_targets = data.loc[:, ["Price"]]
"""

"""
3.  How large a dataset do you think you could load this way?
    What might be the limitations?
    Hint: consider the time to read the data, representation, processing, and memory footprint.
    Try this out on your computer. What happens if you try it out on a server?
"""

# This dataset has 45,212 rows and 17 columns, with a size of 4.5 MB.
bank_data_file = os.path.join("..", "data", "bank-full.csv")

read_time_0 = time.perf_counter()
bank_data = pd.read_csv(bank_data_file, sep=";")
read_time_1 = time.perf_counter()
print(f"Time to read {(read_time_1 - read_time_0) * 1000}ms")  # 87.4ms

separation_time_0 = time.perf_counter()
bank_inputs = bank_data.iloc[:, 0:16]
bank_targets = bank_data.iloc[:, 16]
separation_time_1 = time.perf_counter()
print(f"Time to separate inputs and targets {(separation_time_1 - separation_time_0) * 1000}ms")  # 3.58ms

one_hot_time_0 = time.perf_counter()
bank_inputs = pd.get_dummies(bank_inputs, dummy_na=True)
bank_targets = pd.get_dummies(bank_targets, dummy_na=True)
one_hot_time_1 = time.perf_counter()
print(f"Time to one-hot encoding {(one_hot_time_1 - one_hot_time_0) * 1000}ms")  # 25.1ms
print(bank_inputs)

tensor_creation_time_0 = time.perf_counter()
bank_X = torch.tensor(bank_inputs.to_numpy(dtype=float))
bank_y = torch.tensor(bank_targets.to_numpy(dtype=float))
tensor_creation_time_1 = time.perf_counter()
print(f"Time to tensor creation {(tensor_creation_time_1 - tensor_creation_time_0) * 1000}ms")  # 8.8ms
print(bank_X)
print(bank_y)

"""
4.  How would you deal with data that has a very large number of categories?
    What if the category labels are all unique?
    Should you include the latter?
    
For the case of all unique categories, the column should probably be removed.

For cases where there a large number of categories, the correct approach will likely depend on the situation.
    - You could try concatenating shared categories into groups.
    - You could try to ascertain categorical implications to consolidate information into a single, or smaller number of,
    values.
    - You could drop the column.
"""
