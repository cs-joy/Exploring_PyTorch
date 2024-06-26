
# Exercises

## 1. Documentation reading
A big part of deep learning (and learning to code in general) is getting familiar with the documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes reading the following (it's okay if you don't get some things for now, the focus is not yet full understanding, it's awareness):

* The documentation on [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)
* The documentation on [`torch.cuda`](https://pytorch.org/docs/stable/cuda.html)
"""

import torch

# default float datatype is `float32`

default_float_datatype_tensor = torch.tensor([0.5, 0.1, 0.7, 0.3])
default_float_datatype_tensor, default_float_datatype_tensor.dtype

# torch.Tensor

## Data Types

# float32 datatype
# create tensor
float_32_tensor = torch.tensor([4, 7, 8], dtype=torch.float) # `torch.float` or, `torch.float32`
float_32_tensor, float_32_tensor.dtype

float_32Tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
float_32Tensor

# float64 datatype
float_64_tensor = torch.tensor([1, 4, 2], dtype=torch.float64) # `torch.float64` or, `torch.double`
float_64_tensor, float_64_tensor.dtype

float_64Tensor = torch.tensor([6, 9, 2], dtype=torch.double)
float_64Tensor, float_64Tensor.dtype

'''
1. Sometimes referred to as binary16: uses 1 sign, 5 exponent, and 10 significand bits. Useful when precision is important at the expense of range.
'''

# 1. float16 datatype
float_16_tensor = torch.tensor([7, 3, 4], dtype=torch.float16)
float_16_tensor, float_16_tensor.dtype

#
float_16Tensor = torch.tensor([8, 2, 6], dtype=torch.half)
float_16Tensor, float_16Tensor.dtype

'''
Sometimes referred to as Brain Floating Point: uses 1 sign, 8 exponent, and 7 significand bits. Useful when range is important, since it has the same number of exponent bits as float32
'''

# 2. float16 datatype
float_16_tensor = torch.tensor([7, 3, 4], dtype=torch.bfloat16)
float_16_tensor, float_16_tensor.dtype

# default integer datatype is `int64`
default_int_datatype_tensor = torch.tensor([6,1,9])
default_int_datatype_tensor, default_int_datatype_tensor.dtype

# integers(signed) datatypes
int_8_tensor = torch.tensor([4,8,1], dtype=torch.int8)
int_8_tensor, int_8_tensor.dtype

int_16_tensor = torch.tensor([6,8,2], dtype=torch.int16)
int_16_tensor, int_16_tensor.dtype

int_16_or_short_tensor = torch.tensor([7,1,9], dtype=torch.short)
int_16_or_short_tensor, int_16_or_short_tensor.dtype

# 32bit integers (signed)
int_32_tensor = torch.tensor([6,3,9], dtype=torch.int32)
int_32_tensor, int_32_tensor.dtype

int32Tensor = torch.tensor([6,1,9], dtype=torch.int)
int32Tensor, int32Tensor.dtype

int_64_Tensor = torch.tensor([7,1,8], dtype=torch.int64)
int_64_Tensor, int_64_Tensor.dtype

int_64_or_long_Tensor = torch.tensor([7,1,8], dtype=torch.long)
int_64_or_long_Tensor, int_64_or_long_Tensor.dtype

# boolean datatype
boolean_tensor = torch.tensor([1,0,1,0,1], dtype=torch.bool)
boolean_tensor, boolean_tensor.dtype

## backward compatibility
# 32-bit floating point
x = torch.FloatTensor([1,8,5])
x, x.dtype

# converting datatype (float32 to float64)

# 1.
y = x.type(torch.DoubleTensor)
y, y.dtype

# 2.
z = x.type(torch.float64)
z, z.dtype

# 3.
w = x.type(torch.double)
w, w.dtype

# 64-bit floating point
a = torch.DoubleTensor([7,1,5,4])
a

# 16-bit floating point
b = torch.HalfTensor([4,7,1])
b

# 16-bit floating point
c = torch.BFloat16Tensor([5,8,4])
c

# 8-bit integer (unsigned)
d = torch.ByteTensor([1,2,3])
d

# 8-bit integer (signed)
e = torch.CharTensor([3,5,4])
e

# 16-bit integer (signed)
f = torch.ShortTensor([5,7,9])
f

# 32-bit integer (signed)
g = torch.IntTensor([6,1,5])
g

# 64-bit integer (signed)
h = torch.LongTensor([5,2,9])
h

# Boolean
i = torch.BoolTensor([0,1,0])
i, i.dtype

# factory function - `torch.empty()`
k = torch.empty(size=(3,3),dtype=torch.float16)
k, k.ndim, k.shape

# initializing and basic operations
m = torch.tensor([
    [1., -1.],
    [1., -1.]
])
m

import numpy as np

from_numpy = torch.tensor(
    np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
)
from_numpy

check = torch.tensor([1, 2, 3], dtype=torch.float16)
check.requires_grad_()

# another method
check2 = torch.tensor([1.0, 4.1, 9.2])
check2.detach_(), check2.dtype

y = torch.tensor(check2) # To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
y

#
TENSOR_X = torch.tensor([2.1, 6.4, 3.9], requires_grad=True)

TENSOR_Y = TENSOR_X.detach()

TENSOR_Y.requires_grad_()

checkTensor = torch.Tensor([1, 2, 3])
checkTensor, checkTensor.dtype

"""# Tensor class reference
```
class     torch.Tensor
```
There are a few main ways to create a tensor, depending on your use case.

1. To create a tensor with pre-existing data, use `torch.tensor()`.

2. To create a tensor with specific size, use `torch.*` tensor creation ops (see [Creation Ops](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)).

3. To create a tensor with the same size (and similar types) as another tensor, use `torch.*_like` tensor creation ops (see [Creation Ops](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)).

4. To create a tensor with similar type but different size as another tensor, use `tensor.new_*` creation ops.


"""

# 1
first_tensor = torch.tensor([6, 3, 2])
first_tensor

# 2 torch.*

# 2.1 - torch.zeros(dimension or size)
second_tensor = torch.zeros(3, 4)
second_tensor

# 2.2 torch.zeros_like(source_tensor)
source_tensor = torch.tensor([[3, 4, 6], [1, 8, 3]])

third_tensor = torch.zeros_like(source_tensor)
third_tensor

# 2.3 torch.ones(dimension)
ones_tensor = torch.ones(5,8)
ones_tensor

# 2.4 torch.ones_like(source_tensor)
source_tensor = torch.tensor([[3, 6, 7], [9,5,8]])
ones_like_tensor = torch.ones_like(source_tensor)
print(f"source_tensor = \n{source_tensor}")
print(f"ones_like_tensor = \n{ones_like_tensor}")

# 2.5 torch.arange(start, end, step) - returns a 1-D tensor of size [end - start / step]
arange_tensor = torch.arange(1, 15, 2)
arange_tensor

# 2.6 torch.linspace(start, end, step) - Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
linspace_tensor = torch.linspace(60, 100, 14)
linspace_tensor

# 2.7 torch.logspace(start, end, step, base) - Creates a one-dimensional tensor of size steps whose values are evenly spaced from base start base_start to base_end base end, inclusive, on a logarithmic scale with base base.
logspace_tensor = torch.logspace(3, 9, 5) # by default base=10.0
logspace_tensor

'''
calculation for getting the values of the return Tensor,
value = base ** (start + i * (end-start) / (steps-1)),   where `i` is the index of the value in the sequence, ranging from 0 to `steps-1`
'''

logspace_tensor = torch.logspace(0, 2, 10)
logspace_tensor

exploring_logspace = torch.logspace(start=0.1, end=1.0, steps=1)
exploring_logspace

exploring_logspace = torch.logspace(start=0.1, end=1.0, steps=2)
exploring_logspace

exploring_logspace = torch.logspace(start=0.1, end=1.0, steps=3)
exploring_logspace

exploring_logspace = torch.logspace(start=0.1, end=1.0, steps=4)
exploring_logspace

exploring_logspace = torch.logspace(start=0.1, end=1.0, steps=5)
exploring_logspace

'''
calculation of `torch.logspace(start=0.1, end=1.0, steps=10)`

values of the return tensor are:

1st_value  = base ** (start + (steps - 10) * (end - start / steps - 1)) [equivalent of (base ** start), if you put those values in the equation you will understand why it's equivalent with `base ** start`]
2nd_value  = base ** (start + (steps - 9) * (end - start / steps - 1))
3rd_value  = base ** (start + (steps - 8) * (end - start / steps - 1))
4th_value  = base ** (start + (steps - 7) * (end - start / steps - 1))
5th_value  = base ** (start + (steps - 6) * (end - start / steps - 1))
6th_value  = base ** (start + (steps - 5) * (end - start / steps - 1))
7th_value  = base ** (start + (steps - 4) * (end - start / steps - 1))
6th_value  = base ** (start + (steps - 3) * (end - start / steps - 1))
8th_value  = base ** (start + (steps - 2) * (end - start / steps - 1))
10th_value = base ** (start + (steps - 1) * (end - start / steps - 1)) [equivalent of (base ** end), if you put those values in the equation you will understand why it's equivalent with `base ** end`]

notice: for user defined purposes if we mention `i` instead of (steps - 10, 9, 8, 7, 6, 5, 4, 3, 2, 1) that means the index of the `i` will be start `0` to `steps - 1`
        as for this example 0 to 9 because `steps = 10`

source: https://pytorch.org/docs/stable/generated/torch.logspace.html#torch.logspace
'''


exploring_logspace = torch.logspace(start=0.1, end=1.0, steps=10)
exploring_logspace

# 2.8 torch.eye() -> Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
eye_tensor = torch.eye(5) # paramers of eye() are, 1. n (which will be a int number) -> the number of rows 2. m (which will be a int number) -> the number of columns with default being n
eye_tensor, eye_tensor.shape, eye_tensor.ndim

eye_tensor = torch.eye(8, 8)
eye_tensor, eye_tensor.shape, eye_tensor.ndim

# 2.9 - torch.empty(size of the tensor) -> Return a tensor filled with uninitialized data.
empty_tensor = torch.empty((3, 4))
empty_tensor

# 3.0 - torch.empty_like(input_tensor) -> Returns an uninitialized tensor with the same size as input.
input_tensor = torch.empty((3, 4), dtype=torch.int32)

empty_like_tensor = torch.empty_like(input_tensor)
empty_like_tensor

# 3.1 ->

x = torch.rand(3, 100)
x, x.shape, x.ndim

x.stride()

# Create a 2D array
x = torch.tensor(
    [

              [0, 1, 2, 3, 4,7],
              [5, 6, 7, 8, 9,8],

    ], dtype=torch.int32)

x.stride(), x.ndim

x[0][0].item() #x_00

#x_10
x[1][0]

import numpy as np

t = torch.tensor(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32))
t.stride()

x = torch.arange(12).view((3,4))
x

idx = (2, 1)
item = x[idx].item()
item

# torch.squeeze()
x = torch.rand(1,5,3,1,2,1)

print({x}, {x.ndim}, {x.shape})

sq = torch.squeeze(x)
print({sq}, {sq.ndim}, {sq.shape})

# torch.unsqueeze() -> Returns a new tensor with a dimension of size one inserted at the specified position.
us = torch.unsqueeze(sq, 2)
print({us}, {us.shape}, {us.ndim})

"""# Indexing, Slicing, Joining, Mutating Ops["""





"""# [2. Create a random tensor with shape(7,7)](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

# create random tensor
random_tensor = torch.randn(7, 7)
random_tensor

"""# [3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape(1,7) (hint: you may have to transpose the second tensor)](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

second_random_tensor = torch.randn(1, 7)
second_random_tensor

# multiplication
#mul_tensor = torch.matmul(random_tensor, second_random_tensor) # generate error because of shape issue(mat1 and mat2 shapes cannot be multiplied (7x7 and 1x7))
#mul_tensor

# hence we have to transpose second_random_tensor to solve the issue
transpose_of_second_random_tensor = second_random_tensor.T
transpose_of_second_random_tensor

# again try to multiplication
mul_tensor = torch.matmul(random_tensor, transpose_of_second_random_tensor)
mul_tensor

"""#[4. Set the random seed to 0 and do exercises 2 & 3 over again.](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

# set the manual seed
torch.manual_seed(0)

# create two random tensor
X = torch.rand(size=(7, 7))
Y = torch.rand(size=(1, 7))

# Matrix multiply tensors
Z = torch.matmul(X, Y.T)
Z, Z.shape

"""#[5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation for torch.cuda for this one). If there is, set the GPU random seed to `1234`.](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

# set the random seed on the GPU
torch.cuda.manual_seed(1234)

"""# [6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

# set random seed
torch.manual_seed(1234)

# check for access to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# create two random tensors on GPU
tensor1 = torch.rand(2, 3).to(device)
print(tensor1)
tensor2 = torch.rand(2, 3).to(device)
print(tensor2)

"""# [7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

# transpose of second tensor(tensor2)
transpose_of_tensor2 = tensor2.T
print(transpose_of_tensor2.shape)

# multiplication
mul_tensor1_tensor2 = torch.matmul(tensor1, transpose_of_tensor2)
mul_tensor1_tensor2, mul_tensor1_tensor2.shape

"""# [8. Find the maximum and minimum values of the output of exercise no. 7](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

# maximum
max_of_mul_tensor1_tensor2 = torch.max(mul_tensor1_tensor2)
print(max_of_mul_tensor1_tensor2)

# minimum
min_of_mul_tensor1_tensor2 = torch.min(mul_tensor1_tensor2)
print(min_of_mul_tensor1_tensor2)

"""# [9. Find the maximum and minimum index values of the output of exercise no. 7.](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

#
arg_max = torch.argmax(mul_tensor1_tensor2)

arg_min = torch.argmin(mul_tensor1_tensor2)

arg_max, arg_min

torch.manual_seed(1111)
t = torch.rand(5,10,10)
print(t)
print(t.ndim)

print(torch.max(t))
print(torch.min(t))

print(torch.argmax(t))
print(torch.argmin(t))



mul_tensor1_tensor2

"""# [10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises)"""

# create random tensor
torch.manual_seed(7)
r_tensor = torch.rand(1, 1, 1, 10)
print(r_tensor)
print(r_tensor.shape)

# remove the single dimensions
new_tensor = torch.squeeze(r_tensor)
print(new_tensor)
print(new_tensor.shape)



