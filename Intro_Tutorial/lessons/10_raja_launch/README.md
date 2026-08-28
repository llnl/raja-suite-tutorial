# Lesson 10: RAJA Launch

In this lesson, we begin exploring the RAJA abstraction for exposing parallelism within nested loops and the GPU thread/block programming model.

Prior to this lesson, we have only explored the ``RAJA::forall`` method. Although easy to use, the RAJA::forall method is limited to creating
a single parallel region and distributing the iterations of a for loop to threads via OpenMP or a GPU execution policy.

The ``RAJA::launch`` API, as demonstrated below, is designed to create a kernel execution space in which developers can express their algorithms
using nested ``RAJA::loop`` methods. Similar to GPU programming models, a compute grid can be configured using the ``RAJA::Teams`` and ``RAJA::Threads``
constructs within the ``RAJA::LaunchParams`` container.

``RAJA::loop`` methods can be executed sequentially using ``RAJA::seq_exec``. Loop iterations can be dispatched from CUDA threads and blocks using the
policies shown here for the x-dimension (similar policies exist for the y- and z-dimensions). Similar policies exist for other GPU programming models
(HIP/SYCL).

| RAJA                                        |       CUDA                              |
|---------------------------------------------|---------------------------------------- |
| RAJA::cuda_thread_direct_x                  | threadIdx.x                             |
| RAJA::cuda_block_direct_x                   | blockIdx.x                              |
| RAJA::cuda_global_size_x_direct<block_size> | blockIdx.x * blockDim.x  + threadIdx.x  |


```
  RAJA::launch<EXEC_POL>(
    RAJA::LaunchParams(RAJA::Teams(teams), RAJA::Threads(team_size)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        // ``RAJA::loop`` may be nested within an kernel execution space
        RAJA::loop<LOOP_POL>(ctx, RAJA::TypedRangeSegment<int>(0,M), [&] (int row) {
          RAJA::loop<LOOP_POL>(ctx, RAJA::TypedRangeSegment<int>(0,N), [&] (int col) {
              // Computation
          });
        });

    });
```

In this final lesson, we invite the particpants to complete the policy selection for a matrix-transpose example.
