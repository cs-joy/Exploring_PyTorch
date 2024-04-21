# -*- coding: utf-8 -*-

### PyTorch Fundamentals
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

"""### Introduction to Tensors

### Creating tensors

PyTorch tensors are created using `torch.Tensor()` = https://pytorch.org/docs/stable/tensors.html
"""

# scalar
scalar = torch.tensor(7)
scalar

# number of dimension : https://pytorch.org/docs/stable/generated/torch.Tensor.ndim.html

scalar.ndim

# Get tensor back as Python int

scalar.item()

# Vector
vector = torch.tensor([7, 7])
vector

vector.ndim

vector.shape

# MATRIX
MATRIX = torch.tensor([[2,4],
                       [1,3]])
MATRIX

MATRIX.ndim

MATRIX[0]

MATRIX.shape

# Tensor
TENSOR = torch.tensor([[[1,3,9],
                        [2,4,6],
                        [5,9,7]]])
TENSOR

TENSOR.ndim

TENSOR.shape

TENSOR[0]

MY_TENSOR = torch.tensor([[[
                              [1,3,9,4], # <0
                              [2,4,8,6], # <1
#                              [4,1,6,3]  # <2
#                              ^ ^ ^ ^
#                              0 1 2 3
                            ]]])
MY_TENSOR

MY_TENSOR.ndim

MY_TENSOR.shape

"""## Tensor Exercise"""

TENSOR = torch.tensor([[1,2,3], [4,5,6]])
TENSOR

TENSOR.ndim

TENSOR.shape

TENSOR[0]

TENSOR2 = torch.tensor([
    [2,1,4],
    [1,5,2]
])
TENSOR2

TENSOR2.ndim

TENSOR2.shape

TENSOR3 = torch.tensor([
    [2,1,4],
    [1,5,2],
    [7,5,9]
])
TENSOR3

TENSOR3.ndim

TENSOR3.shape

TENSOR4 = torch.tensor([[
    [1,2,3,2],
    [7,8,9,7],
    [4,5,6,9]
]])

TENSOR4

TENSOR4.ndim

TENSOR4.shape

TENSOR5 = torch.tensor(
    [
        [
            [
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ]
                ]
            ]
        ]
    ]
)

TENSOR5

TENSOR5.ndim # expected 6

TENSOR5.shape #expected tensor.Size([1,1,1,1,3,2])

TENSOR6 = torch.tensor(
    [
            [
              [1,2],
              [4,8],
              [3,9]
            ],
            [
              [5,6],
              [7,9],
              [4,1]
            ]

    ]
)

TENSOR6

TENSOR6.ndim

TENSOR6.shape

TENSOR7 = torch.tensor(
    [
        [
            [
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ]
            ],
            [
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ]
            ]
        ],
                [
            [
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ]
            ],
            [
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ],
                [
                    [
                        [1,2],
                        [4,1],
                        [6,8]
                    ],
                    [
                        [4,1],
                        [1,5],
                        [47,9]
                    ]
                ]
            ]
        ]
    ]
)
TENSOR7

TENSOR7.ndim

TENSOR7.shape

"""### Random Tensors

Why random tensors?

Random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data.

`Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers`

Torch random tensors = https://pytorch.org/docs/stable/generated/torch.rand.html
"""

# Create a random tensors of size (3,4)
random_tensors = torch.rand(3, 4)
random_tensors

random_tensors.ndim

random_tensors = torch.rand(10, 8)
random_tensors

random_tensors.ndim

random_tensors = torch.rand(10, 100, 100)
random_tensors

random_tensors.ndim

random_tensors = torch.rand(10, 10, 10, 10, 10)
random_tensors

random_tensors.ndim

random_tensors.shape

# Create a random tensor with similar shape to an image tensor
random_image_tensor = torch.rand(size=(224, 224, 3)) # height, width, colour channels (R, G, B)
random_image_tensor.shape, random_image_tensor.ndim

itensor = torch.rand(224, 224, 3)
itensor

itensor.ndim

"""## Zeros and ones"""

# Create a tensor of all zeros
zeros = torch.zeros(size=(3,4))
zeros

new_random_tensor = torch.rand(3,4)
new_random_tensor

zeros * new_random_tensor

# Create a tensor of all ones
ones = torch.ones(size=(3,4))
ones

ones.dtype

"""## Creating a range of tensors and tensors-like

`torch.arange()` : https://pytorch.org/docs/stable/generated/torch.arange.html
"""

# Use torch.range(1,11)
# Depreciated message:::>>>> UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).

one_to_ten = torch.arange(1,11)
one_to_ten

zero_to_hundred = torch.arange(start=0, end=100, step=10)
zero_to_hundred

# Creating tensors like
# when we have to define a tensor shape from a pre-defined tensor and overwrite with zeros
new_tensor_from_zero_to_hundred_tensor = torch.zeros_like(input=zero_to_hundred)
new_tensor_from_zero_to_hundred_tensor

"""## Tensor Datatypes

**Note:** Tensor datatypes is one the 3 big errors you'll run into with PyTorch && Deep Learning:

1.   Tensors not right datatype
2.   Tensors not right shape
3.   Tensors not on the right device


Precision in computing: https://en.wikipedia.org/wiki/Precision_(computer_science)


"""

# default tensor datatype - default tensor datatype is depends on input input of the given tensor

# here i will set `int` numbers for the tensor, so the default datatype of the tensor will be `int64`
# so the default is `int64` for int numbers
default_datatype_tensor = torch.tensor([5,8,1,6,2,9])

default_datatype_tensor

default_datatype_tensor.dtype

# here i will set `float` numbers for the tensor, so the default datatype would be `float32`
# so the default is `float32` for int numbers
default_datatype_tensor = torch.tensor([3.1,4.9,6.4,.5])

default_datatype_tensor

default_datatype_tensor.dtype

# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 2.0, 4.0], dtype=torch.float32) # defined `float32`

float_32_tensor

float_32_tensor.dtype

# set float numbers but later define the datatype your want to sent with the tensor
int64_tensor = torch.tensor([4.1,2,4,7], dtype=torch.int64)
int64_tensor

int64_tensor.dtype

#
float_32_tensor = torch.tensor([7.3,1,9,4.2], dtype=None # what datatype is the tensor (e.g. float32. float16)
                                                        )
float_32_tensor

float_32_tensor.dtype

# with more parameters
float_32_tensor = torch.tensor([4.1,7.5,7.8],
                               dtype = None, # set  datatypes:  e.g. float16, float32 :> available datatypes: https://pytorch.org/docs/stable/tensors.html || https://en.wikipedia.org/wiki/Precision_(computer_science)
                               device = None, # what device is your tensor on e.g. 'cpu' or 'cuda' mean gpu
                               requires_grad = False # whether or not to track gradients with this tensors operations
                               )
float_32_tensor

# converted the datatype 'float_32' to 'float_16' of a tensor

# how might we change the datatype of the tensor
float16_tensor = float_32_tensor.type(torch.float16)
float16_tensor

"""# Excercise (DataTypes)"""

# boolean data types
boolean_tensor = torch.tensor([1,0,0,1,0,1,1], dtype=torch.bool)

boolean_tensor

boolean_tensor.dtype

# requires_grad = True
MY_TENSOR = torch.tensor([[1.1, 9.3], [3.6, 6.7]], requires_grad=True)

OUT = MY_TENSOR.pow(2).sum()
OUT.backward()
MY_TENSOR.grad

"""## Tensor Views
PyTorch allows a tensor to be a `View` of an existing tensor. View tensor shares the same underlying data with its base tensor. Supporting `View` avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.

For example, to get a view of an existing tensor `first_tensor`, you can call `t.view(...)`.


**Source:** https://pytorch.org/docs/stable/tensor_view.html#tensor-view-doc
"""

# example
first_tensor = torch.rand(4,4)
first_tensor

second_tensor = first_tensor.view(2,8)
second_tensor

# check if the two tensors sharing the same data

# using `first_tensor.storage()` you will get deprecated message, To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
first_tensor.untyped_storage().data_ptr() == second_tensor.untyped_storage().data_ptr() # `first_tensor` and `second_tensor` share the same underlying data

# Modifying view tensor(e.g.`second_tensor`) changes base tensor(e.g. `first_tensor`) as well

# first extract value of given indices
second_tensor[0][0]

# updating value
second_tensor[0][0] = 0.401

# check whether the value of the first_tensor is updated or not
first_tensor[0][0]

# print the whole tensor
first_tensor

#    Note:** Since views share underlying data with its base tensor, if you edit the data in the view, it will be reflected in the base tensor as well.

"""Typically a PyTorch op returns a new tensor as output, e.g. `add()`. But in case of view ops, outputs are views of input tensors to avoid unnecessary data copy. No data movement occurs when creating a view, view tensor just changes the way it interprets the same data. Taking a view of contiguous tensor could potentially produce a non-contiguous tensor. Users should pay additional attention as contiguity might have implicit performance impact. `transpose()` is a common example.

**Source:** https://pytorch.org/docs/stable/tensor_view.html#tensor-views
"""

first_tensor.is_contiguous()

second_tensor.is_contiguous()

# create new base tensor
base_tensor = torch.tensor([[0,1],[2,3]])
print(base_tensor.is_contiguous())
base_tensor

view_tensor = base_tensor.transpose(0,1) # `view_tensor` if a view of `base_tensor`. No data movement happened here
view_tensor

# `view_tensor`(transpose) might be non-contiguous
# so, let's check whether it's right or not
view_tensor.is_contiguous()

# To get contiguous tensor, call `.contiguous()` to enfore
# copying data when `view_tensor` is not contiguous
copy_tensor = view_tensor.contiguous()
copy_tensor

# check now, if `copy_tensor` is contiguous or not
copy_tensor.is_contiguous()

"""## Arithmetic Operations"""

# multiplication with two different datatypes or with same datatype but different bits(e.g. 16 or 32 or 64)

float_16_tensor = torch.tensor([1.1,5.3,6.9], dtype=torch.float16)

float_16_tensor

flotat_32_tensor = torch.tensor([4.7, 4.1,8.3], dtype=torch.float32)

float_32_tensor

float_16_tensor * float_32_tensor

int_32_tensor = torch.tensor([5,8,3], dtype=torch.int32)
int_32_tensor

int_32_tensor * float_32_tensor

int_64_tensor = torch.tensor([2,7,9], dtype=torch.int64)
int_64_tensor

float_32_tensor * int_64_tensor

long_tensor = torch.tensor([4,6,9], dtype=torch.long)
long_tensor

long_tensor * float_32_tensor

"""## Getting information from tensors (tensor attributes)
1.   Tensors not right datatype - to do that get datatype from a tensor, can use `tensor.dtype`
2.   Tensors not right shape - to get shape from a tensor, can use `tensor.shape`
3.   Tensors not on the right device - to get device from a tensor, can use `tensor.device`
"""

# Create a tensor
some_tensor = torch.rand(3,4)
some_tensor

# Find out details anout some tensor
print(some_tensor)
print(f"DataType of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}") # there are two way to get shape or size of a tensor, 1. with attribute (e.g. some_tensor.shape) 2. with function (e.g. some_tensor.size())
print(f"Device of tensor is on: {some_tensor.device}")

check = torch.rand(3,4)
check

check.dtype

deck = check.type(torch.float16)
deck.dtype

deck

"""### Manipulating Tensors (tensor operations)

Tensor operations include:
* Addition
* Subtraction
* Multiplication (element-wise)
* Division
* Matrix Multiplication
"""

# Create a tensor and add 10 to it
tensor = torch.tensor([1,2,3])
tensor + 10

# Subtract 10
tensor - 10

# Multiply tensor by 10
tensor * 10

tensor

# Try out PyTorch in-built functions
torch.mul(tensor, 10)

torch.add(tensor,10)

"""### Matrix multiplication (Part #1)

Two main ways of performing multiplication in neural networks and deep learning.

1. Element-wise Multiplication
2. Matrix Multiplication (dot prouct)

More information on multiplying matrices - https://www.mathsisfun.com/algebra/matrix-multiplying.html#
"""

# Element wise multiplication
print(tensor, "*", tensor)
print(f"Equals: {tensor * tensor}")

# Matrix Multiplication by torch
torch.matmul(tensor, tensor)

# Matrix Multiplication by hand
1*1 + 2*2 + 3*3

# Commented out IPython magic to ensure Python compatibility.
# %%time
# value = 0
# for i in range(len(tensor)):
#   value += tensor[0] * tensor[0]
# print(value)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# torch.matmul(tensor, tensor)

"""### Matrix Multiplication (Part #2)

There are two main rules that performing matrix multiplication needs to stisfy:
1.  The **inner dimensions** must match:
    *   `(3,2) @ (3,2)` won't work
    *   `(2,3) @ (3,2)` will work
    *   `(3,2) @ (2,3)` will work

    ***Note:*** `@` is a attribute in tensor for multiplying two tensor whereas `tensor.matmul()` is a function. e.g. `tensor1 @ tensor2` `==` `torch.matmul(tensor1, tensor2)`
2.  The resulting matrix has the shape of the **outer dimensions**:
    *   `(2,3) @ (3,2)` -> `(2,2)`
    *   `(3,2) @ (2,3)` -> `(3,3)`

#### One of the main errors in matrix multiplication is shape error
"""

torch.matmul(torch.rand(3,2), torch.rand(3,2)) # error occurs

torch.matmul(torch.rand(2,3), torch.rand(3,2))

torch.matmul(torch.rand(3,2), torch.rand(2,3))

torch.matmul(torch.rand(4,5), torch.rand(5,4))

torch.matmul(torch.rand(10,4), torch.rand(4, 5))

# if inner dimension match but different outer dimension
torch.matmul(torch.rand(10,4), torch.rand(4, 6))

"""### One of the most common errors in deep learning: shape errors

### Matrix Multiplication (Part #3: Dealing with tensor shape errors)
"""

# Shapes for matrix multiplication
tensor_A = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])
tensor_B = torch.tensor([[7,10],
                         [8,11],
                         [9,12]])

#torch.mm(tensor_A, tensor_B) # torch.mm is the same as torch.matmul (it's as an alias for writing less code)
torch.matmul(tensor_A, tensor_B)

tensor_A.shape, tensor_B.shape

"""To fix our tensor shape issues, we can manipulate the shape of one of our tensors using **transpose**.

**Note:** A **transpose** switches the axes or dimensions of a given tensor
"""

tensor_B, tensor_B.shape

tensor_B.t() # transpose via function

tensor_B.T, tensor_B.T.shape #tranpose via attribute

# second approach to make transpose through function
torch.transpose(tensor_B, 0, 1) # reference:: https://pytorch.org/docs/stable/generated/torch.transpose.html

# The matrix multiplication operation works when tensorB is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}")
print(f"Multiplying: {tensor_A.shape} @ {tensor_B.T.shape} <- inner dimensions must match")
print(f"Output: \n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")

"""### Finding the min, max, mean, sum etc (tensor aggregation)"""

# Create a tensor
x = torch.arange(0, 100, 10)
x, x.dtype

# Find the min
torch.min(x), x.min()

# Find the max
torch.max(x), x.max()

# Find the mean **note: the torch.mean() function requires a tensor of float32 datatype to work
torch.mean(x.type(torch.float32)), x.type(torch.float32).mean()

# Find the sum
torch.sum(x), x.sum()

"""### Finding the positional min and max of tensors"""

x = torch.arange(1, 100, 10)
x

# Find the position in tensor that has the minimum value with `argmin()` -> returns index position of target tensor where the minimum value occurs.
x.argmin()

# check the value of the position
print(f"x.argmin().item() = {x.argmin().item()}\n")
x[x.argmin().item()]

# Find the position in tensor that has the maximum value with `argmax()`
x.argmax()

# check the value of the position
print(f"x.armax().item() = {x.argmax().item()}\n")
x[x.argmax().item()]

