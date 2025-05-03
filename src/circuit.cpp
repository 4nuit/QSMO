/**
 * @file circuit.cpp
 * @brief Quantum circuit simulation using SYCL for parallel computation on various devices (GPU, FPGA, CPU).
 */

#include "../headers/circuit.hpp"
#include <chrono>

using namespace sycl;

/**
 * @brief Constructs a quantum circuit with a specified number of qubits.
 *
 * Initializes the state vector in the |00...00> state and allocates memory on the device.
 * Also selects the appropriate device (GPU, FPGA, or CPU) based on preprocessor definitions.
 *
 * @param numQubits The number of qubits in the quantum circuit
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
Circuit::Circuit(size_t numQubits, bool use_single_task)
    : numQubits(numQubits), use_single_task(use_single_task), selector(), queue(selector, properties)
{
    // Initialisation state |00...00> - 1-qbit to |0>
    std::fill(stateVector, stateVector + numStates, Complex<float>(0.0f, 0.0f));
    stateVector[0] = Complex<float>(1.0f, 0.0f);

    stateVector_d = malloc_device<Complex<float>>(numStates, queue);

    // Create queue selector and commands
    auto device = queue.get_device();
    std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
}

/**
 * @brief Destructor for the Quantum Circuit.
 *
 * Measures the total elapsed time since the circuit was created and frees the allocated
 * device memory for the state vector.
 */
Circuit::~Circuit()
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total elapsed time: " << duration.count() << "ms\n";
    sycl::free(stateVector_d, queue);
    delete[] stateVector;
    stateVector_d = nullptr;
    stateVector = nullptr;
}

/**
 * @brief Applies the Hadamard gate to the specified qubit.
 * The Hadamard gate is applied to the target qubit, which puts it into a superposition state.
 * @param target The qubit index to which the Hadamard gate will be applied
 */
void Circuit::h(int target)
{
    ::h(queue, stateVector_d, numQubits, target, use_single_task);
}

/**
 * @brief Applies the Hadamard gate to a group of n qubit.
 * @param n The number of consecutive qubit indeesx to which the Hadamard gate will be applied
 */
void Circuit::h_n(int n)
{
    ::h_n(queue, stateVector_d, numStates);
}

/**
 * @brief Applies the X gate to the specified qubit.
 * The X gate applies a NOT on the considered qubut.
 * @param target The qubit index to which the X gate will be applied
 */
void Circuit::x(int target)
{
    ::x(queue, stateVector_d, numQubits, target, use_single_task);
}

/**
 * @brief Applies the Y gate to the specified qubit.
 * @param target The qubit index to which the Y gate will be applied
 */
void Circuit::y(int target)
{
    ::y(queue, stateVector_d, numQubits, target, use_single_task);
}

/**
 * @brief Applies the Z gate to the specified qubit.
 * The Z gate is applied to the target qubit, flipping its phase.
 * @param target The qubit index to which the Z gate will be applied
 */
void Circuit::z(int target)
{
    ::z(queue, stateVector_d, numQubits, target, use_single_task);
}

/**
 * @brief Applies the Rx gate to the specified qubit.
 * The Rx gate is applied to the target qubit, flipping its phase.
 * @param target The qubit index to which the Rx gate will be applied
 */
void Circuit::rx(int target)
{
    ::rx(queue, stateVector_d, numQubits, target, angle, use_single_task);
}

/**
 * @brief Applies the Ry gate to the specified qubit.
 * The Ry gate is applied to the target qubit, flipping its phase.
 * @param target The qubit index to which the Ry gate will be applied
 */
void Circuit::ry(int target)
{
    ::ry(queue, stateVector_d, numQubits, target, angle, use_single_task);
}

/**
 * @brief Applies the Rz gate to the specified qubit.
 * The Rz gate is applied to the target qubit, flipping its phase.
 * @param target The qubit index to which the Rz gate will be applied
 */
void Circuit::rz(int target)
{
    ::rz(queue, stateVector_d, numQubits, target, angle, use_single_task);
}

/**
 * @brief Applies the CNOT gate to the 2 specified qubits.
 * The CNOT applies a NOT on the 2nd qubit if the first is |1>.
 * @param target1 The first qubit on which the CNOT gate is applied.
 * @param target2 The second qubit on which the CNOT gate is applied.
 */
void Circuit::cnot(int target1, int target2)
{
    ::cnot(queue, stateVector_d, numQubits, target1, target2, use_single_task);
}

/**
 * @brief Applies the SWAP gate to the 2 specified qubits.
 * @param target1 The first qubit on which the SWAP gate is applied.
 * @param target2 The second qubit on which the SWAP gate is applied.
 */
void Circuit::swap(int target1, int target2)
{
    ::swap(queue, stateVector_d, numQubits, target1, target2, use_single_task);
}

/**
 * @brief Applies the Toffoli gate to the 3 specified qubits.
 * @param target1 The first qubit on which the Toffoli gate is applied.
 * @param target2 The second qubit on which the Toffoli gate is applied.
 * @param target3 The third qubit on which the Toffoli gate is applied.
 */
void Circuit::ccnot(int target1, int target2, int target3)
{
    ::toffoli(queue, stateVector_d, numQubits, target1, target2, target3, use_single_task);
}

/**
 * @brief Applies the Controlled Phase Flip gate to the 3 specified qubits.
 * @param target1 The first qubit on which the Toffoli gate is applied.
 * @param target2 The second qubit on which the Toffoli gate is applied.
 * @param target3 The third qubit on which the Toffoli gate is applied.
 */
void Circuit::ccflip(int target1, int target2, int target3)
{
    ::controlledPhaseFlip(queue, stateVector_d, numQubits, target1, target2, target3, use_single_task);
}

/**
 * @brief Measures the quantum state and samples it multiple times.
 * This method performs a measurement on the quantum state and samples the outcome.
 * @param numSamples The number of samples to measure from the quantum state
 */
void Circuit::measure(int numSamples)
{
    ::measure(queue, stateVector_d, numQubits, numSamples, use_single_task);
}

// Gate enum for the circuit

/**
 * @brief Constructs a quantum circuit with a specified number of qubits.
 *
 * Initializes the state vector in the |00...00> state and allocates memory on the device.
 * Also selects the appropriate device (GPU, FPGA, or CPU) based on preprocessor definitions.
 *
 * Gates are computing using oneMath SYCL BLAS kernels.
 *
 * @param numQubits The number of qubits in the quantum circuit
 */
BlasCircuit::BlasCircuit(size_t numQubits) : Circuit::Circuit(numQubits, false)
{
    std::fill(stateVector, stateVector + numStates, 0.0f);

    y = malloc_device<float>(numStates, queue);
    queue.memcpy(y, stateVector, numStates * sizeof(float)).wait();

    stateVector[0] = 1.0f;
    stateVector_d = malloc_device<float>(numStates, queue);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(float)).wait();
}

/**
 * @brief Applies the Hadamard gate to the specified qubit.
 * The Hadamard gate is applied to the target qubit, which puts it into a superposition state.
 * @param target The qubit index to which the Hadamard gate will be applied
 */
void BlasCircuit::h(int target)
{
    ::blas_h(queue, stateVector_d, numQubits, alpha, beta, target);
}

/**
 * @brief Applies the X gate to the specified qubit.
 * The X gate applies a NOT on the considered qubut.
 * @param target The qubit index to which the X gate will be applied
 */
void BlasCircuit::x(int target)
{
    ::sparse_x(queue, stateVector_d, numQubits, alpha, beta, target);
}

/**
 * @brief Applies the X gate to ALL qubits within the stateVector.
 */
void BlasCircuit::x_all()
{
    ::sparse_x_all(queue, stateVector_d, y, numQubits, alpha, beta);
}

/**
 * @brief Applies the Z gate to the specified qubit.
 * The Z gate is applied to the target qubit, flipping its phase.
 * @param target The qubit index to which the Z gate will be applied
 */
void BlasCircuit::z(int target)
{
    ::sparse_z(queue, stateVector_d, numQubits, alpha, beta, target);
}

/**
 * @brief Applies the CCNOT gate to ALL qubits within the stateVector.
 */
void BlasCircuit::ccnot_all()
{
    ::sparse_ccnot_all(queue, stateVector_d, numQubits, alpha, beta);
}

/**
 * @brief Applies the Controlled Phase Flip gate to the 3 specified qubits.
 * @param target1 The first qubit on which the Toffoli gate is applied.
 * @param target2 The second qubit on which the Toffoli gate is applied.
 * @param target3 The third qubit on which the Toffoli gate is applied.
 */
void BlasCircuit::ccflip(int target1, int target2, int target3)
{
    ::sparse_ccflip(queue, stateVector_d, numQubits, alpha, beta, target1, target2, target3);
}

/**
 * @brief Measures the quantum state and samples it multiple times.
 * This method performs a measurement on the quantum state and samples the outcome.
 * @param numSamples The number of samples to measure from the quantum state
 */
void BlasCircuit::measure(int numSamples)
{
    ::blas_measure(queue, y, numQubits, numSamples);
}
