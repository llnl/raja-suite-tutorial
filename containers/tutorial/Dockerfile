###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

FROM ghcr.io/llnl/radiuss:cuda-13-0-ubuntu-24.04

ARG USER=rajadev
ARG CUDA_HOME=/usr/local/cuda-13.0
ARG CUDA_ARCHITECTURES=90
ARG njobs=2
ENV HOME /home/${USER}

RUN sudo apt-get update && sudo apt-get install -y supervisor
RUN sudo useradd --create-home --shell /bin/bash ${USER}

WORKDIR /opt/archives
RUN sudo curl -L https://github.com/gitpod-io/openvscode-server/releases/download/openvscode-server-v1.69.1/openvscode-server-v1.69.1-linux-x64.tar.gz > \
    /opt/archives/openvscode-server-v1.69.1-linux-x64.tar.gz
RUN sudo tar xzf openvscode-server-v1.69.1-linux-x64.tar.gz && sudo chown -R ${USER}:${USER} openvscode-server-v1.69.1-linux-x64

USER ${USER}

COPY --chown=rajadev:rajadev . $HOME/tutorial/

RUN  mkdir -p ${HOME}/tutorial/caliper_build && cd ${HOME}/tutorial/caliper_build && \
              cmake -DCUDAToolkit_ROOT=${CUDA_HOME} -DWITH_NVTX=ON -DWITH_CUPTI=ON -DCMAKE_INSTALL_PREFIX=$(pwd) ../tpl/caliper/ && \
              make install -j"${njobs}"

ENV CALIPER_PATH  ${HOME}/tutorial/caliper_build

RUN mkdir -p ${HOME}/tutorial/build && cd ${HOME}/tutorial/build && \
    cmake -DCMAKE_CXX_COMPILER=g++ \
          -DCUDAToolkit_ROOT=${CUDA_HOME} \
          -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
          -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc \
          -DENABLE_CUDA=On \
          -DBLT_CXX_STD=c++20 \
          "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
          -DCMAKE_CUDA_FLAGS="--extended-lambda --expt-relaxed-constexpr" \
          -DENABLE_OPENMP=On \
          -DRAJA_ENABLE_CALIPER=ON \
          -Dcaliper_DIR=${CALIPER_PATH}/share/cmake/caliper \
          -DCMAKE_BUILD_TYPE=Release \
          .. \
    && make -j"${njobs}" RAJA \
    && make -j"${njobs}" umpire \
    && make -j"${njobs}"

USER root
ADD ./containers/tutorial/supervisord.conf /etc/supervisord.conf
RUN sed -i "s/XXX/${USER}/g" /etc/supervisord.conf

RUN touch /var/log/openvscode-server.log && chown -R ${USER}:${USER} /var/log/openvscode-server.log

CMD ["/usr/bin/supervisord"]
