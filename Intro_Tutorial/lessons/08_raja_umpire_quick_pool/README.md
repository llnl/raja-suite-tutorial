# Lesson 8: Umpire Memory Pools

In this lesson, you will learn to create and use an Umpire memory pool.

Frequently allocating and deallocating memory can be quite costly, especially
when you are making large allocations or allocating on different memory
resources. Memory pools are a more efficient way to allocate large amounts of
memory, especially in HPC environments. Below is a visual representation of a
memory pool which contains multiple "chunks" or "blocks" of memory. The pool is
accessible by the underlying Allocator that it is built on top of.

<figure>
<img src="./images/Umpire-Picture1.png">
</figure>

Umpire provides **allocation strategies** (such as memory pools) that can be 
used to customize how data is obtained from the system. In this lesson, we 
will learn about one such strategy called `QuickPool`. 

The `QuickPool` strategy describes a certain type of pooling algorithm provided 
by Umpire. As its name suggests, `QuickPool` is performant for many use cases. 

We recommend users start with `QuickPool`, but Umpire also provides other 
types of pooling strategies such as `DynamicPoolList` and `FixedPool`. 
More information about Umpire memory pools and other features
is available in the [Umpire User Guide](https://umpire.readthedocs.io/en/develop/index.html).

To create a new memory pool allocator using the `QuickPool` strategy, we use
the `ResourceManager`:
```
  umpire::Allocator pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool_name", my_allocator);
```

This newly created `pool` is an `umpire::Allocator` that uses the `QuickPool`
allocation strategy. In the code example above, we call the
`ResourceManager::makeAllocator` function to create the pool allocator. We
pass in: (1) the name we choose for the pool, and (2) an allocator we
previously created with the `ResourceManager`. Note that you will need to
include the Umpire header file for the pool type you wish to use, in this case
```
#include "umpire/strategy/QuickPool.hpp"
```

When you have created your QuickPool allocator, uncomment the COMPILE definition on line 7;
then compile and run the code:
```
$ make 08_raja_umpire_quick_pool
$ ./bin/08_raja_umpire_quick_pool
```

Other arguments can be passed to the pool constructor if needed. However, they
are beyond the scope of this tutorial. Please visit the [Umpire User Guide](https://umpire.readthedocs.io/en/develop/index.html) to learn more. 

