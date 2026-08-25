# Lesson 4: RAJA Simple Loops

Data parallel kernels are common in many parallel HPC applications. In a data
parallel loop kernel, the processing of data that occurs at each iterate **is
independent** of the processing of data at all other iterates. This is
sometimes referred to as "embarrassingly parallel" because it is
straightforward to parallelize a kernel when there is no chance that the
computation done in one thread or process can impact the computation done in
another.

A simple example of a data parallel loop kernel that is parallelized using
OpenMP is:

```
double data* = new....;

#pragma omp parallel for
for (int i = 0; i < N; ++i) {
  data[i] = i;
}
```

Each loop iterate sets the array element at the iterate index to the index
value. Clearly, each iterate is independent of the others. If this OpenMP
kernel were run with M threads, then depending on how the loop work is
scheduled, iterates may be partitioned into chunks of size N/M with each 
thread executing one chunk of iterates. This is illustrated in the figure.

<figure>
<img src="./images/parchunk.png">
</figure>

If the loop takes T time units to run on one process/thread, then ideally it
would run on T / M time units in parallel when using M processors/threads
(M <= N). However, parallel overheads often prevent one from observing this
optimal speed up. Indeed, depending on the kernel, number of iterates, number
of threads, etc., a kernel may run slower in parallel than it does sequentially.

In this lesson, you will learn about the `RAJA::forall` loop kernel execution
method to parallelize this kernel.

The `RAJA::forall` template method is specialized on an execution policy type
parameter that specifies how the kernel will be compiled to run. A
`RAJA::forall` method takes two arguments: an iteration space object,
such as a contiguous range of loop indices as shown in this lesson, and a
C++ lambda expression that represents the loop kernel body:

```
RAJA::forall<EXEC_POLICY>( ITERATION SPACE, LAMBDA);
```

To describe an iteration space that is a contiguous sequence of integers
`[0, N)`, we create a `RAJA::TypedRangeSegment` as follows:

```
RAJA::TypedRangeSegment<int>(0, N)
```

The lambda expression takes one argument, the loop iterate index:

```
[=](int i) { // loop body }
```

The `[=]` syntax tells the lambda to capture arguments by value (e.g. create a
copy, rather than a reference).

The code for this lesson resides in the file `04_raja_forall.cpp`. It provides
a RAJA implementation of a kernel that sets each element of an array `data` to
the value of its array index using the `RAJA::seq_exec` policy. With this
policy, the loop will execute sequentially on a CPU. The code will record the
time of the loop execution and print it out along with a few values of the
array to show that the array entries are set as expected. 

Following that, you will see a `TODO` comment where you can add a similar
`RAJA::forall` kernel to set the elements of the array `data1` in the same way
as the sequential kernel. However, you will use an OpenMP execution policy
`RAJA::omp_parallel_for_exec` to run the loop in parallel on a CPU. Again, the
code will record and print the kernel execution time and array values for
comparison to the previous case and verification that they are set as you
expect.

When you have made your changes, compile and run the code in the same way as the
other lessons:

```
$ make 04_raja_forall
$ ./bin/04_raja_forall
```

If you need help, you can compare your version of the code to the solution
code using the command `diff 04_raja_forall.cpp 04_raja_forall_solution.cpp`.

Are the array elements that are printed out the same in each case? How do the 
execution times compare? Which kernel ran faster?

For more information about `RAJA::forall` use, execution policies, etc. please
see [RAJA Basic Loop Execution](https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/add_vectors.html).

