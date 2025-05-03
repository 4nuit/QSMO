#include <iostream>
#include <sycl/sycl.hpp>
#include "../headers/circuit.hpp"

using namespace sycl;

void multiControlledPhaseFlip(Circuit& circuit, int numQubits)
{

  // Apply multiple CCZ gates in order to flip only the last qbit
  for (int q = 2; q < numQubits; q++)
  {
    circuit.ccflip(q - 2, q - 1, q);
  }

  for (int q = numQubits - 2; q >= 2; q--)
  {
    circuit.ccflip(q, q - 1, q - 2);
  }
}

/*
 * effect |solElem> -> -|solElem> via a
 * multi-controlled phase flip gate
 */
void applyOracle(Circuit& circuit, int numQubits, int solElem)
{

  // apply X to transform |111> into |solElem>
  for (int q = 0; q < numQubits; q++){
    if (((solElem >> q) & 1) == 0){
      circuit.x(q);
    }
  }

  // effect |111> -> -|111>
  multiControlledPhaseFlip(circuit, numQubits);

  // apply X to transform |solElem> into |111>
  for (int q = 0; q < numQubits; q++){
    if (((solElem >> q) & 1) == 0){
      circuit.x(q);
    }
  }
}

/* apply 2|+><+|-I by transforming into the Hadamard basis
 * and effecting 2|0><0|-I. We do this, by observing that
 *   c..cZ = diag{1,..,1,-1}
 *         = I - 2|1..1><1..1|
 * and hence
 *   X..X c..cZ X..X = I - 2|0..0><0..0|
 * which differs from the desired 2|0><0|-I state only by
 * the irrelevant global phase pi
 */
void applyDiffuser(Circuit& circuit, int numQubits)
{

  // apply H to transform |+> into |0>
  for (int q = 0; q < numQubits; q++){
    circuit.h(q);
  }

  // apply X to transform |11..1> into |00..0>
  for (int q = 0; q < numQubits; q++){
    circuit.x(q);
  }

  // effect |11..1> -> -|11..1>
  multiControlledPhaseFlip(circuit, numQubits);

  // apply X to transform |00..0> into |11..1>
  for (int q = 0; q < numQubits; q++){
    circuit.x(q);
  }

  // apply H to transform |0> into |+>
  for (int q = 0; q < numQubits; q++){
    circuit.h(q);
  }
}

/** @brief Apply the main algorithm using kernels.cpp gates behind circuit.cpp class.
 *
 */

int main()
{
  bool passed = true;
  try
  {
    // Create a Circuit object
    constexpr size_t numQubits = 5;
    bool use_single_task = false;

    // Initialisation state |00...00>
    Circuit circuit(numQubits, use_single_task);

    // randomly choose the element for which to search
    int numStates = (int) std::pow(2, numQubits);
    int numReps = std::ceil((M_PI/4) * std::sqrt(numStates));
    srand(time(NULL));
    int solElem = rand() % numStates;

    // Initialisation state |+><+| for Grover
    for (int q = 0; q < numQubits; q++){
      circuit.h(q);
    }

    // apply Grover's algorithm
    for (int r = 0; r < numReps; r++)
    {
      applyOracle(circuit, numQubits, solElem);
      applyDiffuser(circuit, numQubits);
    }

    // Measure the states with random sampling
    circuit.measure(100);
    std::cout << "Element to search: " << solElem << std::endl;
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
