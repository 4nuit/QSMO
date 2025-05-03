#include <iostream>
#include <sycl/sycl.hpp>
#include "../../headers/circuit.hpp"

using namespace sycl;


/** @brief Apply the main algorithm using kernels.cpp gates behind circuit.cpp class.
 *
 */

int main()
{
  bool passed = true;
  try
  {
    // Select the device for SYCL execution (e.g., CPU, GPU, FPGA)
    constexpr size_t numQubits = 4;
    bool use_single_task = false;

    // Create a Circuit object
    Circuit circuit(numQubits, use_single_task);   //|0000>

    circuit.x(1);                 //|0010>
    circuit.swap(1,2);            //|0100>

    // Measure the states with random sampling
    circuit.measure(100);
  }
  catch (exception const &e)
  {
    // Catches exceptions in the host code.
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
