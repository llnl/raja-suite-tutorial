if (ENABLE_CUDA)
  if (RAJA_ENABLE_CALIPER)
    tutorial_add_executable(
      NAME profile_raja
      SOURCES profile_raja.cpp
      DEPENDS_ON RAJA umpire cuda)
  endif()
endif()
