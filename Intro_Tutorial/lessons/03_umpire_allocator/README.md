# Lesson 3: Umpire Allocators

In this lesson, you will learn how to use Umpire to allocate memory. The file
`03_umpire_allocator.cpp` contains `TODO:` comments where you will code to
allocate and deallocate memory.

The fundamental concept for accessing memory through Umpire is the
`umpire::Allocator`. An `umpire::Allocator` is a C++ object that can be used to
allocate and deallocate memory, as well as query a pointer to get
information about it. In this lesson, we will see how to query the name of an Allocator.

All `umpire::Allocator` objects are created and managed by the
`umpire::ResourceManager` *Singleton* object. To create an allocator,
first obtain a handle to the ResourceManager, and then request the Allocator
corresponding to the desired memory resource using the `getAllocator` function:

```
auto& rm = umpire::ResourceManager::getInstance();
auto allocator = rm.getAllocator("HOST");
```

The Allocator class provides methods for allocating and deallocating memory. You
can view these methods in the [Umpire AllocatorInterface](https://umpire.readthedocs.io/en/develop/doxygen/html/classumpire_1_1Allocator.html).

To use an Umpire allocator, use the following code, replacing "size in bytes"
with the desired size for your allocation:

```
void* memory = allocator.allocate(size in bytes);
```

Moving and modifying data in a heterogeneous memory system can be subtle
because you have to keep track of the source and destination memory spaces,
and often use vendor-specific APIs to perform the modifications. In Umpire,
all data modification and movement, regardless of memory resource or platform,
is done using **Umpire Operations**.

Next, we will use the `memset` operator provided by Umpire's Resource Manager
to set the memory we just allocated to zero.

Don't forget to deallocate your memory afterwards!

For more details, you can check out the [Umpire Allocator Documentation](https://umpire.readthedocs.io/en/develop/sphinx/tutorial/allocators.html).

Once you have made your changes, you can compile and run the lesson:

```
$ make 03_umpire_allocator
$ ./bin/03_umpire_allocator
Allocated 800 bytes and set to 0 using the HOST allocator.
```
