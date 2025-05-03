#!/bin/bash -l
#SBATCH --time=05:00
#SBATCH --account=lxp
#SBATCH --partition=fpga
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --disable-perfparanoid

module load env/release/2024.1
module load CMake jemalloc ncurses/5.9
module load intel-oneapi
module load 520nmx/20.4

LD_PRELOAD=$(jemalloc-config --libdir)/libjemalloc.so.$(jemalloc-config --revision) srun ./bench/release/grover_5qbits/grover_sycl.fpga_parallel
