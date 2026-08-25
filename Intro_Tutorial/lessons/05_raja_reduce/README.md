# Lesson 5: RAJA Reductions 

In lesson 4, we looked at a data parallel loop kernel in which each loop
iterate was independent of the others. In this lesson, we consider a kernel 
that is **not data parallel** because there is a data dependence between
iterates. A data dependence occurs when multiple threads (or tasks) could
write to the same memory location at the same time. This is often called a
**race condition** and can cause an algorithm to produce **non-deterministic**,
order-dependent results. 

Consider an attempt to write an OpenMP parallel kernel to compute the sum of
the elements in an array:

```
double sum = 0;

#pragma omp parallel for
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
```

This kernel has a race condition since multiple loop iterates could write
to the `sum` variable at the same time. As a result, the computed `sum` value
could be correct, but probably not. OpenMP provides a reduction clause that
allows a user to write a kernel to compute a reduction without a race condition.
That is, only one thread will write to the reduction variable at a time. For
example:

```
double sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; ++i) {
  sum += data[i];
}
```

This kernel implementation does not have a race condition. However, the results
could still be non-deterministic due to the order in which the values are
accumulated in the sum in parallel.

It is important to note that not all parallel programming models provide a
mechanism to write a parallel reduction. For example, CUDA and HIP do not.
RAJA provides a reduction construct enabling users to write parallel reductions
in each of its supported programming model back-ends in a manner that is
portable across those back-ends. This lesson uses the `RAJA::ReduceSum` type
to compute a sum reduction. More information on RAJA reductions, including
other reduction operations that RAJA supports, can be found in the
[RAJA Reduction Operations](https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/reduction.html).

In this lesson, we will use a `RAJA::ReduceSum` object to approximate $\pi$,
the ratio of the circle's circumference to its diameter, using a Riemann integral
approximation of the formula
```math
\frac{ \pi }{4} = \tan^{-1}(1) = \int_0^1 \frac{1}{1+x^2}\,dx \approx \sum_{i=0}^{N} \frac{1}{1 + ( (i+0.5) \Delta x )^{2}} \Delta x
```
where $\Delta x = \frac{1}{N}$.

We create a sum reduction variable `pi` as follows:

```
RAJA::ReduceSum<REDUCTION_POLICY, TYPE> pi(0.0);
```

The `REDUCTION_POLICY` type specializes the reduction object for use in a
kernel with a compatible execution policy. For example, if a `RAJA::forall`
method uses an execution policy `RAJA::omp_parallel_for_exec`, then the
reduction policy must be `RAJA::omp_reduce`. Similarly, for CUDA, HIP, etc.
More information about compatible reduction and execution policies can be
found in [RAJA Reduction Policies](https://raja.readthedocs.io/en/develop/sphinx/user_guide/feature/policies.html#reducepolicy-label).
The second parameter `TYPE` is the data type of the reduced quantity,
such as `int` or `double`. The `0.0` passed to the reduction variable
constructor initializes the sum to zero.

The code for this lesson resides in the file `05_raja_reduce.cpp`. It provides
a RAJA implementation of a kernel that approximates the value of $\pi$ using
the `RAJA::seq_exec` policy. With this policy, the loop will execute
sequentially on a CPU. The code will print out the result which should look
familiar `3.145...` as expected. 

After that, you will find a `TODO` asking you to implement an OpenMP version of
the kernel. You will use `RAJA::omp_parallel_for_exec` for the execution policy
to specialize the `RAJA::forall` template and `RAJA::omp_reduce` for the reduction
policy to specialize the `RAJA::ReduceSum` object.

Once you have filled in the correct reduction statement, compile and run:

```
$ make 05_raja_reduce
$ ./bin/05_raja_reduce
```

Are the results printed for the sequential and OpenMP kernels the same?
Why or why not?

Try running the code multiple times by repeatedly executing:

```
$ ./bin/05_raja_reduce
```

What do you observe about the result printed for the sequential kernel for
each run of the code? What about the results printed for the OpenMP kernel for
each run of the code? Can you explain what is happening?
