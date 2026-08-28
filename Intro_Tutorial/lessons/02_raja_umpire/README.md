# Lesson 2: RAJA and Umpire as Build Dependencies

In this lesson, you will learn how to add RAJA and Umpire as dependencies 
to your application.

RAJA and Umpire are included in this project as **targets** that we tell CMake
our application depends on: [RAJA and Umpire Dependencies](https://github.com/LLNL/raja-suite-tutorial/blob/main/tpl/CMakeLists.txt).

Additionally, we can specify other dependency targets, such as CUDA, in the
`blt_add_executable` macro for our application executable. The macro has
an argument for this, `DEPENDS_ON`, that you can use to list dependencies.

```
blt_add_executable(
    NAME 02_raja_umpire
    SOURCES 02_raja_umpire.cpp
    DEPENDS_ON )
```

In the `CMakeLists.txt` file in this lesson, you will find a `TODO:` comment
asking you to add the RAJA, umpire, and cuda dependencies to build the lesson 
code. After you have added the dependencies, uncomment the RAJA and Umpire
header file includes in the source code. Then, you can build and run the lesson.
As a reminder, open the VSCode terminal (Shift + ^ + `), and then
move to the build directory: 

```
$ cd build
``` 

You can then compile and run the lesson:

```
$ make 02_raja_umpire
$ ./bin/02_raja_umpire
Hello, world (with RAJA and Umpire)!
```

In the next lesson, we will start writing some code!
