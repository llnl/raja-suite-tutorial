macro(tutorial_add_executable)
  set(options)
  set(singleValueArgs NAME TEST REPRODUCER BENCHMARK)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  raja_add_executable(${ARGN})

  if (ENABLE_CUDA)
    target_compile_options(${arg_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  endif()
endmacro()

add_subdirectory(lessons)
