# -*- coding: utf-8 -*-
"""00_pytorch_fundametals_mrdbourke.ipynb


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

"""# Finding the min, max, mean, sum etc (tensor aggregation)"""

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

"""# Finding the positional min and max of tensors"""

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

"""# Reshaping, viewing, stacking, squeezing and unsqueezing tensors

* Reshaping - reshapes an input tensor to a defined shape
* View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
* Stacking - Combine multiple tensors on top of each other (vstack) or side by side (hstack)
    * stack - https://pytorch.org/docs/stable/generated/torch.stack.html
    * vstack - https://pytorch.org/docs/stable/generated/torch.vstack.html
    * hstack - https://pytorch.org/docs/stable/generated/torch.hstack.html
* Squeeze - removes all `1` dimensions from a tensor
* Unsqueeze - add a `1` dimension to a target tensor
* Permute - Return a view of the input with dimension permuted (swapped) in a certain way
"""

#Let's create a tensor

import torch
x = torch.arange(1., 10)
x, x.shape

# Add an extra dimension
x_reshaped = x.reshape(1, 7)
x_reshaped, x_reshaped.shape

x_reshaped = x.reshape(1, 9)
x_reshaped, x_reshaped.shape

x_reshaped = x.reshape(2, 9)
x_reshaped, x_reshaped.shape

x_reshaped = x.reshape(0, 9)
x_reshaped, x_reshaped.shape

x_reshaped = x.reshape(9, 1)
x_reshaped, x_reshaped.shape

# exercise
y = torch.arange(1., 15)
y, y.shape

y_reshaped = y.reshape(2,7) # here,, 2x7 or 7x2 (any other combination of two number) = size of the y tensor(which is 14) *required to match
y_reshaped, y_reshaped.shape

# Change the view
z = x.view(1, 9)
z, z.shape

# Change z changes x (because a view of a tensor shares the same memory as the original input)
z[:, 0] = 5
z, x

# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)
x_stacked

x_stacked = torch.stack([x, x, x, x], dim=1)
x_stacked

x_stacked = torch.stack([x, x, x, x], dim=2)
x_stacked

"""# Squeezing, unsqueezing and permuting tensors
`torch.squeeze()` - removes all single dimensions from a target tensor

For example, if input is of shape: (A×1×B×C×1×D)(A×1×B×C×1×D) then the input.squeeze() will be of shape: (A×B×C×D)(A×B×C×D).

When dim is given, a squeeze operation is done only in the given dimension(s). If input is of shape: (A×1×B)(A×1×B), squeeze(input, 0) leaves the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the shape (A×B)(A×B).

**note:** *The returned tensor shares the storage with the input tensor, so changing the contents of one will change the contents of the other.*

source: https://pytorch.org/docs/stable/generated/torch.squeeze.html
"""

a_reshaped = x_reshaped.reshape(1,9)
a_reshaped

a_reshaped.shape

a_reshaped.squeeze()

a_reshaped.squeeze().shape

#  removes all of the 1 from the input tensor
eg_tensor = torch.rand(3,1,4,2,6, 1, 8)
eg_tensor

eg_tensor.shape

eg_tensor_squeezed = eg_tensor.squeeze()

eg_tensor_squeezed.shape  ## eg_tensor().squeeze() removes all of the 1 from the input tensor

print(f"Previous tensor: {eg_tensor}")
print(f"Previous shape: {eg_tensor.shape}")

print(f"New tensor: {eg_tensor.squeeze()}")
print(f"New shape: {eg_tensor.squeeze().shape}")

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim(dimension)
print(f"Previous target: {eg_tensor_squeezed}")
print(f"Previous shape:{eg_tensor_squeezed.shape}")

eg_tensor_unsqueezed = eg_tensor_squeezed.unsqueeze(dim=0)

print(f"New tensor: {eg_tensor_unsqueezed}")
print(f"New shape: {eg_tensor_unsqueezed.shape}")

# exercise with `my_tensor.squeeze()`

# create tensor
my_tensor = torch.tensor(
    [
        [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
    ]
)
my_tensor

my_tensor.shape, my_tensor.ndim

print(f"Previous tensor:\n {my_tensor}")
print(f"Previous shape: {my_tensor.shape}")
print(f"Previous dimension: {my_tensor.ndim}")

my_tensor_squeezed = my_tensor.squeeze()

print(f"\nLater tensor:\n {my_tensor_squeezed}")
print(f"Later shape: {my_tensor_squeezed.shape}")
print(f"Later dimension: {my_tensor_squeezed.ndim}")

# exercise with my_tensor_squeezed.unsqueeze()

print(f"Previous tensor:\n {my_tensor_squeezed}")
print(f"Previous shape: {my_tensor_squeezed.shape}")
print(f"Previous dimension: {my_tensor_squeezed.ndim}")

my_tensor_unsqueezed = my_tensor_squeezed.unsqueeze(dim=0)

print(f"\nLater tensor:\n {my_tensor_unsqueezed}")
print(f"Later shape: {my_tensor_unsqueezed.shape}")
print(f"Later dimension: {my_tensor_unsqueezed.ndim}")

# torch.permute - rearrage the dimensions of a target tensor in a specified order
# permute is use for work with image data

x_original = torch.rand(size=(224, 224, 3)) # [height, widht, colour_channels]

# Permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}") # [colour_channels, height, width]

x_original[0, 0, 0], x_permuted[0, 0, 0]

x_original[0, 0, 0] = .4507
x_permuted[0, 0, 0]

"""# Indexing (selecting data from tensors)

Indexing with PyTorch is similar to index with NumPy
"""

# Create a tensor

import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape, x.ndim

# Let's index on our new tensor
x[0]

# Let's index on the middke brcket (dim=1)
x[0][0]  # "x[0, 0]" is same as "x[0][0]"

# Let's index on the most inner bracket (last dimension)
x[0][0][0]

x[0][1][1]

# challange to find number 9 from the "x" tensor
x[0][2][2]

# ":" to select "all" of a target dimension
x[:, 0]

x[0,2,1]

t = torch.rand(5, 3, 6, 2)
t

t.ndim, t.shape

t[4]

t.shape

t

t[0]

t[1]

t[2]

t[3]

t[3][2]

t[3][2][5], t[3, 2, 5]

t[3][2][5][1], t[3, 2, 5, 1]

t[:, 2, 5, 1]

t[:3, 2, 5, 1]

t[3:, 2, 5, 1]

x

# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
x[:, :, 1]

# Get all values of the 0th dimension but only the 1 index value of 1st and 2nd dimension
x[:, 1, 1]

# Get index 0 of 0th and first dimension and all values of 2nd dimension
x[0, 0, :]

# Index on x to return 9
x[0, 2, 2]

# Index on x to return `3,6,9`
x[:, :, 2]

"""# PyTorch tensors and NumPy

NumPy is a popular scientific Python numerical computing library

And because of this, PyTorch has functionality to interact with it.

* Data in NumPy, want int PyTorch tensor -> `torch.from_numpy(ndarray)`
* PyTorch tensor -> NumPy -> `torch.Tensor.numpy()`

**learn more:** https://pytorch.org/tutorials/beginner/examples_tensor/polynomial_numpy.html
"""

# NumPy array to tensor
import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # warning: when converting from numpy -> pytorch, pytorch reflects numpy's default datatype of float64 unless specified otherwise
array, tensor

array.dtype # NumPy default datatype

torch.arange(1.0, 8.0).dtype # torch default datatype

# here we convert the datatype `float64` to `float32`

tensor = torch.from_numpy(array).type(torch.float32)
array, tensor, array.dtype, tensor.dtype

# Change the value of the `array`, what will this do to `tensor`?
array = array + 1
array, tensor

# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor, numpy_tensor

numpy_tensor.dtype

# Change the value of the `tensor`, what will this do to `numpy_tensor`
tensor = tensor + 1
tensor, numpy_tensor

"""# Reproducibility (trying to take random out of random)

In short how a neural network learns:

`start with random numbers -> tensor operations -> update random numbers to try and make them better representations of the data -> again -> again -> again...`

To reduce the randomness in neural networks and PyTorch comes the concept of a **random seed**.

Essentially what the random seed does is "flavour" the randomness.

**Learn more:** https://pytorch.org/docs/stable/notes/randomness.html
"""

import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# Let's make some random but reproducible tensors
import torch

# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

"""# Running tensors and PyTorch objects on the GPUs (and making faster computations)

GPUs = faster computation on numbers, thanks to CUDA + NVIDIA hardware + PyTorch working behind the scenes to make everything hunky dory (good).

**Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning::->** https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/
"""

!nvidia-smi

"""## 1. Check for GPU access with PyTorch"""

# Check for GPU access with PyTorch

import torch
torch.cuda.is_available()

"""For PyTorch since it's capable of running compute on the GPU or CPU, it's best practice to setup device agnostic code: https://pytorch.org/docs/stable/notes/cuda.html

**e.g:** run on GPU if available, else default to CPU
"""

# Setup device agnostic code || https://pytorch.org/docs/stable/notes/cuda.html
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Count the number of devices
torch.cuda.device_count()

"""## 3. Putting tensors (and models) on the GPU

The reason we want our tensors/models on the GPU is becaue using GPU results in faster computations.
"""

# Create a tensor (default on the GPU)
tensor = torch.tensor([1, 2, 3], device = "cpu")

# Tensor not on GPU
print(tensor, tensor.device)

# even if we don't declare before
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device) # expected output device is `cpu` because we already write a condition before for the set-uping device

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu

"""### 4. Moving tensors back to the CPU"""

# If tensor is on GPU, can't transform it to NumPy
tensor_on_gpu.numpy()

# To fix the GPU tensor with NumPy issue, we can first get it to the CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu

tensor_on_gpu

