## Reduce Area below kernels occupancy limit for FPGA

These are the same kernels (except for CCFLIP) as `src/kernels.cpp`, but unecessary functions have been removed in order to minimise their size on FPGA (Area <= 100%)

```bash
cd ..

cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/h.cpp;fpga_fix_kernels/circuit.cpp;fpga_fix_kernels/kernels_parallel.cpp" -B build-fpga_emu; cmake --build build-fpga_emu -j$(nproc) --target fpga_emu

cmake -S . -DSOURCE_FILES="algos-examples/gate_tests/h.cpp;fpga_fix_kernels/circuit.cpp;fpga_fix_kernels/kernels_single_task.cpp" -B build-fpga_emu; cmake --build build-fpga_emu -j$(nproc) --target fpga_emu
```
