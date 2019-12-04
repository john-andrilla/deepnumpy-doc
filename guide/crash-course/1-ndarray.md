
# Step 1: Manipulate data with NumPy on MXNet
:label:`crash_course_ndarray`

This getting started exercise introduces the `np` package, which is the primary tool for storing and
transforming data on MXNet. If you’ve worked with NumPy before, you’ll notice `np` is, by design, similar to NumPy.

## Import packages and create an array


To get started, run the following commands to import the `np` package together with the NumPy extensions package `npx`. Together, `np` with `npx` make up the NumPy on MXNet front end.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()  # Activate NumPy-like mode.
```

In this step, create a 2D array (also called a matrix). The following code example creates a matrix with values from two sets of numbers: 1, 2, 3 and 4, 5, 6. This might also be referred to as a tuple of a tuple of integers.

```{.python .input  n=2}
np.array(((1,2,3),(5,6,7)))
```

You can also create a very simple matrix with the same shape (2 rows by 3 columns), but fill it with 1s.

```{.python .input  n=3}
x = np.ones((2,3))
x
```

You can create arrays whose values are sampled randomly. For example, sampling values uniformly between -1 and 1. The following code example creates the same shape, but with random sampling.

```{.python .input  n=15}
y = np.random.uniform(-1,1, (2,3))
y
```

As with NumPy, the dimensions of each ndarray are shown by accessing the `.shape` attribute. As the following code example shows, you can also query for `size`, which is equal to the product of the components of the shape. In addition, `.dtype` tells the data type of the stored values.

```{.python .input  n=17}
(x.shape, x.size, x.dtype)
```

## Performing operations on an array

An ndarray supports a large number of standard mathematical operations. Here are three examples. You can perform element-wise multiplication by using the following code example.

```{.python .input  n=18}
x * y
```

You can perform exponentiation by using the following code example.

```{.python .input  n=23}
np.exp(y)
```

You can also find a matrix’s transpose to compute a proper matrix-matrix product by using the following code example.

```{.python .input  n=24}
np.dot(x, y.T)
```

## Indexing an array

The ndarrays support slicing in many ways you might want to access your data. The following code example shows how to read a particular element, which returns a 1D array with shape `(1,)`.

```{.python .input  n=25}
y[1,2]
```

This example shows how to read the second and third columns from `y`.

```{.python .input  n=26}
y[:,1:3]
```

This example shows how to write to a specific element.

```{.python .input  n=27}
y[:,1:3] = 2
y
```

You can perform multi-dimensional slicing, which is shown in the following code example.

```{.python .input  n=28}
y[1:2,0:2] = 4
y
```

## Converting between MXNet ndarrays and NumPy ndarrays

You can convert MXNet ndarrays to and from NumPy ndarrays, as shown in the following example. The converted arrays do not share memory.

```{.python .input  n=29}
a = x.asnumpy()
(type(a), a)
```

```{.python .input  n=30}
np.array(a)
```

## Next steps

Learn how to construct a neural network with the Gluon module: :ref:`crash_course_nn`.
