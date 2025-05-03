## QSMO

SYCL-based quantum simulator using sparse matrices and qubit wise multiplication for gate computation

- **SpMV** kernels: see [oneapi/math/sparse_bkas/types.hpp](https://github.com/uxlfoundation/oneMath/blob/develop/include/oneapi/math/sparse_blas/types.hpp). **COO** format and **default_alg** with **sorted_by_rows** are used here (*symmetric* is not yet implemented for cuBlas, other algos such as **coo_alg1** could be used).

This project has been tested for Intel CPUs, Nvidia A-100 GPUs and Bittware Stratix 10 FPGAs. You may edit flags (e.g compute capability, targets) in [CMakeLists.txt](CMakeLists.txt) for other targets.

## Credits

This work is the continuation of **FQSIM** for the qubit wise implementation (in `Circuit` class).
See https://github.com/LuxProvide/QuantumFPGA for further details.

## SOTA

### General

- [Simulating Quantum Computers Using OpenCL](https://arxiv.org/pdf/1805.00988)
- [A Herculean task: Classical simulation of quantum computers](https://arxiv.org/pdf/2302.08880)
- [Hybrid Techniques for Simulating Quantum Circuits using the Heisenberg Representation](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/107198/hjgarcia_1.pdf)
- [Architectures and Optimisations for FPGA-based simulation of quantum circuits](https://theses.gla.ac.uk/84894/4/2024MoawadPhD.pdf) #notably Qubit Wised Mult. (cf QCGPU), FPGA optim (kernels, compiler features) (cf OneAPI-Samples git.)

### Optimisations

- [A Systematic Literature Surver of Sparse Matrix-Vector Multiplication](https://arxiv.org/abs/2404.06047)
- [The landscape of software for Tensor Computations](https://export.arxiv.org/pdf/2103.13756)
- [The ITensor software library for tensor network calculations](https://www.scipost.org/SciPostPhysCodeb.4/pdf)
- [Tensor Networks for Simulating Quantum Circuits on FPGAs](https://arxiv.org/pdf/2108.06831)
- [ZX-calculus for the working quantum computer scientist](https://arxiv.org/pdf/2012.13966)

### Libraries, Tools and Techniques 

- https://www.quantiki.org/wiki/list-qc-simulators
- https://www.itensor.org/
- https://github.com/cda-tum/mqt-core

## Dependancies (+Local installation of 1API)

- https://www.intel.com/content/www/us/en/developer/articles/containers/oneapi-base-toolkit.html
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/fpga-download.html?operatingsystem=linux&linux-install=offline
- https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-2/use-the-setvars-and-oneapi-vars-scripts-with-linux.html

```bash
echo -ne ". /opt/intel/oneapi/setvars.sh --force\n. /opt/intel/oneapi/2025.0/oneapi-vars.sh  --force" >> .bashrc && source ~/.bashrc
apt update && apt install libgtest-dev #googletest freeglut3-dev
```

**Celerity (SYCL-MPI)** => Re-implement kernels

```bash
module load intel-compilers/2024.2.0
module load OpenMPI/5.0.3-GCC-13.3.0
# module load CUDA/12.6.0		=> may fail with get_native_access when building
module load CMake/3.29.3-GCCcore-13.3.0 
```

```bash
git clone --recurse-submodules https://github.com/celerity/celerity-runtime
cd celerity-runtime
mkdir install build && cd build
cmake -G "Unix Makefiles" -S .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_INSTALL_PREFIX=~/celerity-runtime/install -DCMAKE_BUILD=Release
make -j$(nproc) install
```

Exemple

```bash
cp -r ~/celerity-runtime/examples/matmul/ matmul && cd matmul
cmake -S . -DCMAKE_CXX_COMPILER=icpx -DCMAKE_INSTALL_PREFIX=~/celerity-runtime/install -DCMAKE_BUILD=Release -B build
cmake --build build -j$(nproc) --target install
./build/executable
```

**GoogleTest**

**oneMath** => Tofix for Tensors

- https://uxlfoundation.github.io/oneMath/using_onemath_with_cmake.html

**MQTCore**  => Will not be used

- https://mqt.readthedocs.io/projects/core/en/latest/installation.html#integrating-mqt-core-into-your-project

## Documentation

The documentation can be generated with:

```bash
doxygen Doxyfile
```

You can create a quantum circuit with different constructors:

```c
Circuit circuit(numQubits,false);   // use parallel_for kernels for the quantum gates
Circuit circuit(numQubits,true);   // use single_task kernels for the quantum gates
BlasCircuit circuit(numQubits);     // use oneMath's BLAS kernels (backends are defined in CMakeLists.txt)
```

## Compilation

1. Load the modules

```bash
module load env/release/2024.1
module load CMake jemalloc ncurses googletest
module load CUDA imkl-FFTW
module load intel-oneapi

# FPGA Only
module load 520nmx/20.4
sbatch ./qsmo_fpga.sh
```

2. Build 

Default build is equivalent to `-DCMAKE_BUILD_TYPE=Release`.

```bash
target=cpu #| gpu | fpga_emu | fpga
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/h.cpp;src/circuit.cpp;src/kernels.cpp" -B build-${target}
cmake --build build-${target} -j$(nproc) --target ${target}
```

**Debug/Profiling**:

```bash
target=cpu #| gpu | fpga_emu | fpga
mkdir build-${target} && cd build-${target}
cmake -S ..  -DCMAKE_BUILD_TYPE=Debug -DSOURCE_FILES="../algos-examples/gate_tests/h.cpp;../src/circuit.cpp;../src/kernels.cpp" #-DUSER_FLAGS="-lGL -lGLU -lglut" for freeglut (in env/staging/2023 , when compiling with blochsphere.cpp)
make VERBOSE=3 -j$(nproc) ${target}
```

### CPU

*Intel MKLCPU (default)*

```bash
## Circuit
#target = cpu | gpu | fpga_emu | fpga
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/h.cpp;src/circuit.cpp;src/kernels.cpp" -B build-cpu
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/x_all.cpp;src/circuit.cpp;src/kernels.cpp" -B build-cpu

## BlasCircuit with MKLCPU
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/sparse_x_all.cpp;src/circuit.cpp;src/kernels.cpp" -B build-cpu

# TODO (Removing main and adding kernels in spmv)
#cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/sparse_x_all.cpp;src/circuit.cpp;src/kernels.cpp" -B build-cpu
#cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/blas_h_tofix.cpp;src/circuit.cpp;src/kernels.cpp" -B build-cpu

cmake --build build-cpu -j$(nproc) --target cpu
```

### GPU

Both Generic or cuBLAS backends are available.

*cuBLAS*

```bash
## Circuit
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/h.cpp;src/circuit.cpp;src/kernels.cpp"  \
   -DENABLE_CUBLAS_BACKEND=True  \
   -DENABLE_CUSPARSE_BACKEND=True \
   -DENABLE_MKLCPU_BACKEND=False \
   -DENABLE_MKLGPU_BACKEND=False  \
   -B build-gpu

cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/x_all.cpp;src/circuit.cpp;src/kernels.cpp" \
   -DENABLE_CUBLAS_BACKEND=True  \
   -DENABLE_CUSPARSE_BACKEND=True \
   -DENABLE_MKLCPU_BACKEND=False \
   -DENABLE_MKLGPU_BACKEND=False  \
   -B build-gpu

## BlasCircuit with cuSPARSE

# TODO
# cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/sparse_x_all.cpp;src/circuit.cpp;src/kernels.cpp" -B build-gpu
# -DSOURCE_FILES="algos-examples/gate_tests/blas_h_tofix.cpp;src/circuit.cpp;src/kernels.cpp"

cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/sparse_x_all.cpp;src/circuit.cpp;src/kernels.cpp" \
   -DENABLE_CUBLAS_BACKEND=True  \
   -DENABLE_CUSPARSE_BACKEND=True \
   -DENABLE_MKLCPU_BACKEND=False \
   -DENABLE_MKLGPU_BACKEND=False  \
   -B build-gpu

cmake --build build-gpu -j$(nproc) --target gpu
```

*Tuning Generic BLAS backend*

- https://github.com/uxlfoundation/generic-sycl-components/tree/main/onemath/sycl/blas#cmake-options

To compile against generic backend, edit **GPU_LINK_FLAGS** in [CMakeLists.txt](./CMakeLists.txt).

```bash
## BlasCircuit with cuSPARSE

# TODO (Removing main and adding kernels in spmv)
# -DSOURCE_FILES="algos-examples/gate_tests/blas_h_tofix.cpp;src/circuit.cpp;src/kernels.cpp"

export SB_ENABLE_JOINT_MATRIX=1
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/sparse_x_all.cpp;src/circuit.cpp;src/kernels.cpp" \
  -DENABLE_GENERIC_BLAS_BACKEND=True \
  -DGENERIC_BLAS_TUNING_TARGET=NVIDIA_GPU \
  -DENABLE_CUSPARSE_BACKEND=True \
  -DENABLE_MKLCPU_BACKEND=False \
  -DENABLE_MKLGPU_BACKEND=False \
  -DENABLE_CUBLAS_BACKEND=False \
  -DENABLE_ROCBLAS_BACKEND=False \
  -DENABLE_NETLIB_BACKEND=False \
  -B build-gpu

cmake --build build-gpu -j$(nproc) --target gpu
```

### FPGA

`-reuse-exe` flag in [CMakeLists.txt](./CMakeLists.txt) allows fast host code recompilation within the same `build-fpga` directory.
See [fpga_fix_kernels](./fpga_fix_kernels/).

```bash
## Circuit
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/h.cpp;fpga_fix_kernels/circuit.cpp;fpga_fix_kernels/kernels_parallel.cpp" -B build-fpga_emu; cmake --build build-fpga_emu -j$(nproc) --target fpga_emu

## BlasCircuit with GEMM (no SPARSE backend)
# Doesnt work for now
cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/h.cpp;src/circuit.cpp;src/kernels.cpp" \
  -DENABLE_GENERIC_BLAS_BACKEND=True \
  -DENABLE_MKLCPU_BACKEND=False \
  -DENABLE_MKLGPU_BACKEND=False \
  -DENABLE_CUBLAS_BACKEND=False \
  -DENABLE_ROCBLAS_BACKEND=False \
  -DENABLE_NETLIB_BACKEND=False \
  -B build-fpga_emu

cmake --build build-fpga_emu -j$(nproc) --target fpga_emu
#cmake --build build-fpga -j$(nproc) --target fpga
```

3. Run the program

```bash
# For FPGA targets (DMA with Alignement), and CPU
export JEMALLOC_PRELOAD=$(jemalloc-config --libdir)/libjemalloc.so.$(jemalloc-config --revision)
time LD_PRELOAD=${JEMALLOC_PRELOAD} ./quantum.<target>
```

## Tests

Use single task kernels:

```bash
sed -i -e "s/use_single_task = false/use_single_task = true/g" tests/*cpp
```

```bash
# target = cpu | gpu | fpga_emu
export target=gpu

cmake --build build-${target}/tests -j$(nproc) --target ${target}_h ${target}_x ${target}_y ${target}_z ${target}_rx ${target}_ry ${target}_rz 
cmake --build build-${target}/tests -j$(nproc) --target ${target}_cnot ${target}_swap
cmake --build build-${target}/tests -j$(nproc) --target ${target}_ccnot ${target}_ccflip

# sparse_x_all
cmake --build build-${target}/tests -j$(nproc) --target ${target}_spxa
```

```bash
ctest --test-dir build-${target} -R ${target}_h -j$(nproc) --output-on-failure -V 
ctest --test-dir build-${target} -R ${target}_spxa -j$(nproc) --output-on-failure -V

# run all tests
ctest --test-dir build-${target} -j$(nproc)
```

## Profiling , Benchmarks & Performance debugging

See [bench](./bench)

```bash
module load env/staging/2024.1
module load Linaro-Forge/24.0.6-GCC-13.3.0 GCC GCCcore
forge-probe #gathering architecture details
map --profile --cuda-kernel-analysis ./quantum.gpu
map &       #loading the .map profile
```

**Energy** : 

- GPU: See Power metric in Linaro
- FPGA:

```bash
srun -A lxp -p gpu -q default -N 1 -n 1 --time=01:00 --disable-perfparanoid sbatch bench_gpu.sh
sacct -j <jobid> -o jobid,jobname,partition,account,state,consumedenergyraw
```

```bash
srun -A lxp -p fpga -q default -N 1 -n 1 --time=01:00 --disable-perfparanoid sbatch bench_fpga.sh
sacct -j <jobid> -o jobid,jobname,partition,account,state,consumedenergyraw
```
