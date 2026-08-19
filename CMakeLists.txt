cmake_minimum_required(VERSION 3.23.1)

project(RajaSuiteTutorial LANGUAGES C CXX)

set (ENABLE_TESTS Off CACHE BOOL "")
set (ENABLE_EXAMPLES Off CACHE BOOL "")
set (ENABLE_REPRODUCERS Off CACHE BOOL "")
set (ENABLE_EXERCISES Off CACHE BOOL "")
set (RAJA_ENABLE_EXERCISES Off CACHE BOOL "")
set (ENABLE_DOCUMENTATION Off CACHE BOOL "")
set (ENABLE_BENCHMARKS Off CACHE BOOL "")

set(BLT_CXX_STD "c++20")
include(blt/SetupBLT.cmake)

if (ENABLE_CUDA)
  if (DEFINED CMAKE_CUDA_ARCHITECTURES)
    if ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "70")
      message(STATUS "CUDA compute architecture set to 70")
    else ()
      set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "Set CMAKE_CUDA_ARCHITECTURES to RAJA minimum supported" FORCE)
    endif ()
  endif()
endif()

add_subdirectory(tpl)
add_subdirectory(Intro_Tutorial)
add_subdirectory(Intermediate_Tutorial)
add_subdirectory(Profile_Demo)
