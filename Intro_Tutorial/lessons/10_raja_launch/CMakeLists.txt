if (ENABLE_CUDA)
  tutorial_add_executable(
    NAME 10_raja_launch_transpose
    SOURCES 10_raja_launch_transpose.cpp
    DEPENDS_ON RAJA umpire cuda)

  tutorial_add_executable(
    NAME 10_raja_launch_transpose_solution
    SOURCES 10_raja_launch_transpose_solution.cpp
    DEPENDS_ON RAJA umpire cuda)
endif()
