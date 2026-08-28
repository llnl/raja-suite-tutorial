# Lesson 9: RAJA Views and Layouts

In this lesson, you will learn how to use `RAJA::View` to simplify
multidimensional indexing.

As is commonly done for efficiency in C and C++, we have allocated the data as
one-dimensional arrays. Thus, we need to manually compute the
data pointer offsets for the row and column indices in the kernel.

A `RAJA::View<TYPE, LAYOUT>` type takes two template parameters. The `TYPE`
parameter is the data type of the underlying data. The `LAYOUT` parameter
is a `RAJA::Layout` type that describes how the View indices are ordered
with respect to data access. A two-dimensional RAJA View constructor takes
three arguments; for example, 
```
RAJA::View<TYPE, LAYOUT> my_view(data_ptr, extent0, extent1);
```
The `data_ptr` is a pointer to the data array that the View will be used to
index into. The extent arguments specify the ranges of the indices in each 
dimension. A three-dimensional RAJA View constructor takes four arguments,
a data pointer and three extents, one for each View dimension. And so on for
higher dimensions.

The `RAJA::Layout<DIM, TYPE>` takes two template parameters. The `DIM` parameter
is the number of indexing dimensions, and the `TYPE` parameter is the data type
of the indices used to index into the underlying data. For example, a
two-dimensional layout for a view that takes `int` values to index into the
data is defined as:
```
RAJA::Layout<2, int>
```

It is essential to note that the default data layout ordering in RAJA is
row-major, which is the convention for multi-dimensional array indexing in C
and C++. This means that the rightmost index will be stride-1, the index to
the left of the rightmost index will have stride equal to the extent of the
rightmost dimension, and so on.

Tying everything together, we construct a two-dimensional MxN View that uses
integer indices to access entries in an array of doubles as follows:

```
double* data = ...;
RAJA::View<double, RAJA::Layout<2, int>> view(data, M, N);
```
Note that the size of the data array must be at least MxN to avoid issues.

In the file `09_raja_view.cpp`, there are `TODO` comments asking you to create
two views, `A` and `R`. `A` will be a standard RAJA view and `R` will be 
created using the `RAJA::make_permuted_view` helper method that takes a
right-oriented layout, which is the same as `A`. Knowledge of 
`RAJA::make_permuted_view` is not required to complete this task. If you 
wish to learn more details, please see [RAJA Make Permuted View](https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/view.html#make-permuted-view).

There are additional `TODO` comments asking you to insert bounds of nested
for-loops, and fill in `A` and `R` with their respective index values.
Finally, check the output to make sure that `A`, `R`, and `L` all print out
the same ordering of index values.
When you are ready, uncomment the `COMPILE` macro and compile and run the code:
```
$ make 09_raja_view
$ ./bin/09_raja_view
```

For more information on RAJA Views and Layouts, please see [RAJA Views and Layouts](https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/view_layout.html).



