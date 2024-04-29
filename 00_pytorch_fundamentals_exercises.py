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

exploring_logspace = torch.logspace(start=0.1, end=1.0, steps=10)
exploring_logspace































"""# Indexing, Slicing, Joining, Mutating Ops["""



