// move in ~/QSMO/algos-exemples/gate_tests/

#include <iostream>
#include <sycl/sycl.hpp>
#include "../../headers/circuit.hpp"

using namespace sycl;

int main()
{
  bool passed = true;
  try
  {
    constexpr size_t numQubits = 10; 
    bool use_single_task = false;

    Circuit circuit(numQubits, use_single_task);

    circuit.h(0);

    circuit.measure(100);
  }
  catch (exception const &e)
  {
    std::cerr << "Caught a SYCL host exception:\n"
              << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND)
    {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
