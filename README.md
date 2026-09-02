[comment]: # (#################################################################)
[comment]: # (Copyright 2016-26, Lawrence Livermore National Security, LLC)
[comment]: # (and RAJA project contributors. See the RAJA/LICENSE file)
[comment]: # (for details.)
[comment]: # 
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# <img src="https://github.com/LLNL/RAJA/blob/develop/share/raja/logo/RAJA_LOGO_CMYK_White_Background_large.png" width="128" valign="middle" alt="RAJA"/>

# RAJA Portability Suite Tutorial

Welcome to the RAJA Portability Suite tutorial. In this repo, you will find
tutorials that show how to use RAJA and Umpire capabilities. There is an
introductory tutorial that provides a guided set of lessons to follow to
learn how to get started with the RAJA Portability Suite. There is also an
intermediate tutorial meant for those who would like more advanced hands-on
instruction for RAJA and Umpire. If you are new to the RAJA Portability Suite,
we suggest that you start the intro tutorial before attempting the intermediate
tutorial.

For more detailed information about using RAJA and Umpire, please refer
to the following:

* [RAJA User Guide](https://raja.readthedocs.io)
* [Umpire User Guide](https://umpire.readthedocs.io)

The lessons that you can build and run depend on what is supported on your
platform. Here are some configuration instructions to get you started on 
Livermore Computing machines. On a CPU-only TOSS4 system, you can build the
lessons that use CPU back-ends, such as OpenMP. On a GPU-enabled system, you
can build the lessons that use CPU and GPU back-ends, such as CUDA or HIP.
Other platforms will be similar.

* First
  * Create a *build directory* in the top-level raja-tutorial-suite directory
  * Run *CMake* from the build directory using one of the recipes below
  * Run *make* in the build directory.
  * Tutorial lesson executable files will be located in the *bin* sub-directory
    in your build directory

Note that you need to use a CMake version greater or equal to 3.25.2 and you
need a C++ compiler (e.g., g++) that supports c++20.

* On a CPU-only TOSS4 system:
```
module load cmake/3.25.2
module load gcc/10.3.1
cmake -DCMAKE_CXX_COMPILER=g++ -DBLT_CXX_STD=c++20 -DENABLE_CUDA=Off -DENABLE_OPENMP=On -DRAJA_ENABLE_EXERCISES=Off -DCMAKE_BUILD_TYPE=Release .. 
```

* On a GPU-enabled system:
```
module load cmake/3.25.2
module load gcc/10.3.1
module load cuda/13.1.1
cmake -DBLT_CXX_STD=c++20 -DENABLE_CUDA=On -DENABLE_OPENMP=On -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-13.1.1/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-13.1.1 -DCMAKE_CUDA_FLAGS="--extended-lambda --expt-relaxed-constexpr" -DCMAKE_BUILD_TYPE=Release ..
```

License
-----------

RAJA is licensed under the [BSD 3-Clause license](https://opensource.org/licenses/BSD-3-Clause).

Copyrights and patents in the RAJA project are retained by contributors.
No copyright assignment is required to contribute to RAJA.

Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-689114`  `OCEC-16-063`

For release details and restrictions, please see the information in the
following:
- [RELEASE](https://github.com/LLNL/RAJA/blob/develop/RELEASE)
- [LICENSE](https://github.com/LLNL/RAJA/blob/develop/LICENSE)
- [NOTICE](https://github.com/LLNL/RAJA/blob/develop/NOTICE)
