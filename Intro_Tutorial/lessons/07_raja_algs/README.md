# Lesson 7: RAJA Algorithms

So far, we've looked at RAJA kernel launch methods, where a user passes a kernel
body that defines what an algorithm does at each iterate. RAJA provides
other parallel constructs that implement particular algorithms and operations
that are important in many HPC applications. These include atomic operations,
scans, and sorts. These constructs are available in RAJA for all supported
programming model back-ends. We discuss atomic operations and scans in detail
in this lesson. For more information about RAJA support for sorting algorithms,
please see [RAJA Sort Algorithms](https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/sort.html).

## Atomic Operations

An **atomic operation** allows only one thread or process at a time to modify
data at a memory location. When a thread/process is about to write to
a memory location, the address is "locked" until the write operation is
complete. Then, the lock is released and another thread/process can write to
the memory location. If the memory location is locked when a thread/process is
about to write to it, the thread/process must wait until the lock is released
to do so. Atomic operations prevent potential memory corruption and can be
essential for a parallel algorithm to be correct and avoid race conditions
that we discussed in lesson 5.

In lesson 5, we looked at a kernel that used a `RAJA::ReduceSum` object to 
approximate $\pi$ by computing a Riemann integral sum. The OpenMP parallel
implementation looked like the following:

```
using EXEC_POL   = RAJA::omp_parallel_for_exec;
using REDUCE_POL = RAJA::omp_reduce; 

constexpr double dx{1.0 / N};

RAJA::ReduceSum<REDUCE_POL, double> pi(0.0);

RAJA::forall<EXEC_POL>(RAJA::TypedRangeSegment<int>(0, N),
  [=] (int i) {
    double x = (double(i) + 0.5) * dx;
    pi += dx / (1.0 + x * x);
  });
double pi_val = 4.0 * pi.get();
```

An alternative implementation of this $\pi$ approximation using a RAJA atomic
operation looks like this:

```
using EXEC_POL   = RAJA::omp_parallel_for_exec;
using ATOMIC_POL = RAJA::omp_atomic;

constexpr double dx{1.0 / N};

double* pi_h{nullptr};

auto& rm = umpire::ResourceManager::getInstance();
auto host_allocator = rm.getAllocator("HOST");

pi_h = static_cast<double*>(host_allocator.allocate(1*sizeof(double)));

pi_h[0] = 0.0;

RAJA::forall<EXEC_POL>(RAJA::TypedRangeSegment<int>(0, N),
  [=] (int i) {
    double x = (double(i) + 0.5) * dx;
    RAJA::atomicAdd<ATOMIC_POL>( pi_h, dx / (1.0 + x * x) );
  });
pi_h[0] *= 4.0;

host_allocator.deallocate(pi_h);
```

The main difference between the previous kernel and this one is that in this
one a call to the `RAJA::atomicAdd` method is made to perform the atomic
update to the memory location defined by the pointer `pi_h`. Note that the
atomic method is specialized to be compatible with the programming model used,
OpenMP in this case. This is similar to the specialization of the
`RAJA::ReduceSum` object in the reduction implementation. Also, we use 
an Umpire host allocator to allocate an array of size 1 to hold the value of 
$\pi$. 

In the file `07_raja_atomic.cpp`, you will see `TODO` comments where you can
fill in the implementation of a RAJA CUDA version of the kernel. To compile and
run the code:

```
$ make 07_raja_atomic
$ ./bin/07_raja_atomic
```

Additional information about RAJA atomic operation support can be found in
[RAJA Atomic Operations](https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/atomic_histogram.html).

## Parallel Scan

A **scan operation** is an important building block for parallel algorithms. It
is a key primitive that is used to convert many sequential operations into
parallel implementations. Scans are based on *reduction tree* and
*reverse reduction tree* structures. Scan is an example of a computation that
looks inherently serial, but for which there exist efficient parallel
implementations. A useful reference that describes how parallel scans are used
is [Prefix Sums and Their Applications by Guy E. Blelloch](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).

RAJA provides commonly-applied scan algorithms, with operators that allow
users to specify the comparison operation used in the scan. The
**parallel prefix sum** is the most common scan operation. A prefix sum
operation takes an array as input and produces an output array containing
partial sums of the input array. To illustrate, consider the following code,
which appears in the file `07_raja_scan.cpp`.

```
constexpr int N = 10;

int in[N] = {8,-1,2,9,10,3,4,1,6,7};
int is_out[N] = {};
int es_out[N] = {};

RAJA::inclusive_scan<EXEC_POL>( RAJA::make_span(in, N),
                                RAJA::make_span(is_out, N) );

std::cout << "Output (inclusive): "; 
for (int i = 0; i < N; ++i) }
  std::cout << is_out[i] << "  ";
}
std::cout << std::endl;

RAJA::exclusive_scan<EXEC_POL>( RAJA::make_span(in, N),
                                RAJA::make_span(es_out, N) );

std::cout << "Output (exclusive): "; 
for (int i = 0; i < N; ++i) }
  std::cout << es_out[i] << "  ";
}
std::cout << std::endl;
```

When this code is run, it will produce the following output:
```
Output (inclusive): 8  7  9  18  28  31  35  36  42  49
Output (exclusive): 0  8  7  9  18  28  31  35  36  42
```

The code shows a `RAJA::inclusive_scan` and a `RAJA::exclusive_scan`. Each
scan method is specialized on an execution policy. RAJA scan methods use the
same execution policies as `RAJA::forall` methods. Second, the arguments to
the scan methods are RAJA *span* objects that are made using the 
`RAJA::make_span` helper method. The helper method takes the address of
an array element and the number of elements in the span, `N` in this example.

Note that the output of the scan operations is similar, but not the same.

The *inclusive scan* fills the output array with partial sums of the input
array. Here, the first element of the input array is 8, so the first element
of the output array is 8. The second element of the input array is -1,
so the second element of the output array is 7, which is 8 + (-1). And so on.

The *exclusive scan* produces the same partial sums but shifts them one slot
to the right in the output array. The first element in the output array for
the exclusive scan is `0`, which is the identity of the scan operator. The
scan operator is `+` in this example, which is the default for RAJA scan 
operations when no operator is specified. An equivalent RAJA scan operation 
where the operator is specified is:

```
RAJA::exclusive_scan<EXEC_POL>( RAJA::make_span(in, N),
                                RAJA::make_span(es_out, N),
                                RAJA::operators::plus<int>{} );
``` 
Similarly, one could pass the operator to the `RAJA::inclusive_scan` method
to achieve the same result as in the example code.

RAJA also provides **in-place** scan operations where the scan output is
produced directly in the input array. For example, continuing with the 
example code above:

```
RAJA::inclusive_scan_inplace<EXEC_POL>( RAJA::make_span(in, N) );

std::cout << "Output (inclusive in-place): ";
for (int i = 0; i < N; ++i) {
  std::cout << in[i] << "  ";
}
std::cout << std::endl;
```

This code produces the following output, which is the same as the result in the
output array in the `RAJA::inclusive_scan` above:
```
Output (inclusive in-place): 8  7  9  18  28  31  35  36  42  49
```

RAJA provides operators that can be used with all scan methods, such as
`RAJA::operators::minimum<T>`, `RAJA::operators::maximum<T>`, and
`RAJA::operators::plus<T>`. These operator types are specialized on data type
`T`, which must match the data type of the array(s) you pass to the scan
methods to produce the results you expect. When no operator is specified,
as in the code examples above, the plus operator `RAJA::operators::plus<T>`
is used.

In the file `07_raja_scan.cpp`, you will see `TODO` comments where you will
implement a RAJA scan that uses the `RAJA::operators::minimum<T>` operator
type to run on a CUDA GPU device. To compile and run the code:

```
$ make 07_raja_scan
$ ./bin/07_raja_scan
```

Is the result what you expected it to be? Can you explain why the first value
in the output is what it is?

Additional information about RAJA scan operations can be found in
[RAJA Parallel Scan Operations](https://raja.readthedocs.io/en/develop/sphinx/user_guide/tutorial/scan.html).
