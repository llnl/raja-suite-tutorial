cmake_minimum_required(VERSION 3.30.5)

project(RajaSuiteTutorial LANGUAGES C CXX)

set (ENABLE_TESTS Off CACHE BOOL "")
set (ENABLE_EXAMPLES Off CACHE BOOL "")
set (ENABLE_REPRODUCERS Off CACHE BOOL "")
set (ENABLE_EXERCISES Off CACHE BOOL "")
set (RAJA_ENABLE_EXERCISES Off CACHE BOOL "")
set (ENABLE_DOCUMENTATION Off CACHE BOOL "")
set (ENABLE_BENCHMARKS Off CACHE BOOL "")

set(BLT_CXX_STD "c++20")

if (ENABLE_CUDA)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "Set CMAKE_CUDA_ARCHITECTURES to default 90")
  else()
    message(STATUS "CUDA compute architecture user-defined to ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
endif()

include(blt/SetupBLT.cmake)

add_subdirectory(tpl)
add_subdirectory(Intro_Tutorial)
add_subdirectory(Intermediate_Tutorial)
add_subdirectory(Profile_Demo)
