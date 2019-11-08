# Manipulate data with DeepNumPy
:label:`crash_course_ndarray`

The `np` package is MXNet’s primary tool for storing and
transforming data. If you’ve worked with `NumPy` before, you’ll notice `np`  is,
by design, similar to NumPy. For more general information about NumPy, see the [NumPy website](https://numpy.org/).

## Getting started

To get started, use the following commands to import the `np` and 'npx' packages. The `npx` package is NumPy extensions. Together, `np` and `npx` are the DeepNumPy frontend.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()  # Change MXNet to the numpy-like mode.
```

The next example shows how to create a 2D array (also called a matrix) with values from two sets of numbers: 1, 2, 3 and 4, 5, 6. This might also be referred to as a tuple of a tuple of integers.

```{.python .input  n=2}
np.array(((1,2,3),(5,6,7)))
```

You can also create a very simple matrix with the same shape (2 rows by 3 columns), but fill it with ones.

```{.python .input  n=3}
x = np.ones((2,3))
x
```

You might want to create arrays whose values are sampled randomly. For example, sampling values uniformly between -1 and 1. In the example here, you create the same shape, but with random sampling.

```{.python .input  n=15}
y = np.random.uniform(-1,1, (2,3))
y
```

As with NumPy, the dimensions of each ndarray are accessible by accessing the `.shape` attribute. You can also query its `size`, which is equal to the product of the components of the shape. In addition, `.dtype` tells the data type of the stored values.

```{.python .input  n=17}
(x.shape, x.size, x.dtype)
```

## Operations

The following examples show how to create operations such as an ndarray or exponents.

An ndarray supports a large number of standard mathematical operations. Such as element-wise multiplication.

```{.python .input  n=18}
x * y
```

This example shows exponentiation.

```{.python .input  n=23}
np.exp(y)
```

You can use a matrix’s transpose to compute a proper matrix-matrix product:

```{.python .input  n=24}
np.dot(x, y.T)
```

## Indexing

The ndarrays support slicing in a variety of ways. Here’s an example of reading a particular element, which returns a 1D array with shape `(1,)`.

```{.python .input  n=25}
y[1,2]
```

Read the second and third columns from `y`.

```{.python .input  n=26}
y[:,1:3]
```

Write to a specific element.

```{.python .input  n=27}
y[:,1:3] = 2
y
```

This example shows that multi-dimensional slicing is also supported.

```{.python .input  n=28}
y[1:2,0:2] = 4
y
```

## Converting between MXNet ndarrays and NumPy ndarrays

You can convert MXNet ndarrays to and from NumPy ndarrays. The converted arrays do not share memory.

```{.python .input  n=29}
a = x.asnumpy()
(type(a), a)
```

```{.python .input  n=30}
np.array(a)
```

## Next Steps

Learn how to construct a neural network with the Gluon module: :ref:`crash_course_nn`.
