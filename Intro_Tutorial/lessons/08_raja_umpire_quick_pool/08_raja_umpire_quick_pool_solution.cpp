#include <iostream>

#include "RAJA/RAJA.hpp"
#include "umpire/Umpire.hpp"
// TODO: include the header file for the Umpire QuickPool strategy so you can
// use it in the code below
#include "umpire/strategy/QuickPool.hpp"

//Uncomment to compile
#define COMPILE

int main()
{
#if defined(COMPILE)

  constexpr int N{10000};
  constexpr std::size_t CUDA_BLOCK_SIZE{256};
  double* a{nullptr};
  double* b{nullptr};
  double* a_h{nullptr};
  double* b_h{nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("DEVICE");
  auto host_allocator = rm.getAllocator("HOST");

  // TODO: create an allocator called "pool" using the QuickPool strategy
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("POOL", allocator);

  a = static_cast<double *>(pool.allocate(N*sizeof(double)));
  b = static_cast<double *>(pool.allocate(N*sizeof(double)));
  a_h = static_cast<double *>(host_allocator.allocate(N*sizeof(double)));
  b_h = static_cast<double *>(host_allocator.allocate(N*sizeof(double)));

  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
      a_h[i] = 1.0;
      b_h[i] = 1.0;
    }
  );

  rm.copy(a, a_h);
  rm.copy(b, b_h);

  double dot{0.0};
  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::TypedRangeSegment<int>(0, N),
    [=] RAJA_DEVICE (int i) {
    cudot += a[i] * b[i];
  });

  dot = cudot.get();
  std::cout << "dot = " << dot << std::endl;

  pool.deallocate(a);
  pool.deallocate(b);
  host_allocator.deallocate(a_h);
  host_allocator.deallocate(b_h);

#endif

  return 0;
}
