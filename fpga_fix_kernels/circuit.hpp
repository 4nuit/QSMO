#ifndef CIRCUIT_HPP
#define CIRCUIT_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../headers/kernels.hpp"

/**
 * @class Circuit
 * @brief Represents a quantum circuit with a state vector and operations like Hadamard and Z gate.
 *
 * This class handles the initialization of the quantum state, provides methods for applying
 * quantum gates (such as Hadamard and Z gates), and supports measuring the quantum state.
 * It uses SYCL to enable execution on various devices such as GPUs, FPGAs, or CPUs.
 */
class Circuit
{

protected:
    size_t numQubits;
    size_t numStates = 1 << numQubits; // 2^n states
    float angle = M_PI / 4;
    Complex<float> *stateVector = new Complex<float>[numStates];
    Complex<float> *stateVector_d;
    std::chrono::high_resolution_clock::time_point start_time;

#if FPGA_SIMULATOR
    sycl::ext::intel::fpga_simulator_selector selector;
#elif FPGA_HARDWARE
    sycl::ext::intel::fpga_selector selector;
#elif FPGA_EMULATOR
    sycl::ext::intel::fpga_emulator_selector selector;
#elif GPU
    sycl::gpu_selector selector;
#else
    sycl::cpu_selector selector;
#endif

    sycl::property_list properties{sycl::property::queue::enable_profiling()};
    sycl::queue queue;
    bool use_single_task;

public:
    Circuit(size_t numQubits, bool use_single_task);
    virtual ~Circuit();

    virtual void h(int target);

    virtual void h_n(int n);

    virtual void x(int target);

    virtual void y(int target);

    virtual void z(int target);

    virtual void rx(int target);

    virtual void ry(int target);

    virtual void rz(int target);

    virtual void cnot(int target1, int target2);

    virtual void swap(int target1, int target2);
    
    virtual void ccnot(int target1, int target2, int target3);

    virtual void ccflip(int target1, int target2, int target3);

    virtual void measure(int numSamples);
};

#endif