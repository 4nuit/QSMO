#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --account=lxp
#SBATCH --partition=fpga
#SBATCH --qos=default
#SBATCH --nodes=1

# For MQT-Core
module load env/release/2024.1
module load CMake jemalloc ncurses/5.9 googletest

# Freeglut & graphviz-python (doxygen) for blochSphere Tests
#module load env/staging/2023.1
#module load CMake/3.18.4 jemalloc freeglut ncurses/5.9 Doxygen graphviz-python

module load CUDA imkl-FFTW
module load intel-oneapi

# FPGA
module load 520nmx/20.4

cmake -S . -DSOURCE_FILES="algos-examples/grover.cpp;fpga_fix_kernels/circuit.cpp;fpga_fix_kernels/kernels_parallel.cpp" -B build-fpga; cmake --build build-fpga -j$(nproc) --target fpga
