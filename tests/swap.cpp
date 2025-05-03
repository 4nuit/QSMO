#ifndef TEST_SWAP_CPP
#define TEST_SWAP_CPP

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../headers/kernels.hpp"
#include <gtest/gtest.h>

using namespace sycl;

/**
 * @brief Testing the SWAP gate on the four possible states: |00>, |01>, |10>, and |11>.
 *
 * This test ensures that the SWAP gate gives the expected results for each of these states.
 */
class SWAPGateTest : public ::testing::Test
{
protected:
    /**
     * @brief Setup method that runs before each test case.
     *
     * Initializes the SYCL queue, selects the device, allocates memory for the
     * state vector, and sets up other variables needed for the tests.
     *
     * Device selection is done based on compilation flags:
     * - FPGA_SIMULATOR: FPGA simulator device
     * - FPGA_HARDWARE: FPGA hardware device
     * - FPGA_EMULATOR: FPGA emulator device
     * - GPU: GPU device
     * - Default: CPU device
     */
    void SetUp() override
    {
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
        device dev = sycl::device(selector);
        queue = sycl::queue(dev, property_list{sycl::property::queue::enable_profiling()});
        use_single_task = false;
        auto device = queue.get_device();
        std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;
        stateVector = new Complex<float>[numStates];
        stateVector_d = malloc_device<Complex<float>>(numStates, queue);
    }

    /**
     * @brief Tear down method that runs after each test case.
     *
     * Cleans up allocated memory for the state vector and device memory.
     */
    void TearDown() override
    {
        delete[] stateVector;
        stateVector = nullptr;
        sycl::free(stateVector_d, queue);
        stateVector_d = nullptr;
    }

    sycl::queue queue;
    bool use_single_task;
    Complex<float> *stateVector;
    Complex<float> *stateVector_d;
    const size_t numQubits = 2; // Two qubits for the SWAP gate
    const size_t numStates = 1 << numQubits; // 4 states for 2 qubits
};

/**
 * @brief Test the SWAP gate on the |00> state.
 *
 * This test ensures that the SWAP gate applied to the |00> state results in the expected |00> state.
 */
TEST_F(SWAPGateTest, TestState00)
{
    Complex<float> expectedState[4] = { Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |00>
    stateVector[0] = Complex<float>(1.0f, 0.0f);
    stateVector[1] = Complex<float>(0.0f, 0.0f);
    stateVector[2] = Complex<float>(0.0f, 0.0f);
    stateVector[3] = Complex<float>(0.0f, 0.0f);

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    swap(queue, stateVector_d, numQubits, 0, 1, use_single_task); // Apply SWAP gate on qubits 0 (control) and 1 (target)
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the SWAP gate on the |01> state.
 *
 * This test ensures that the SWAP gate applied to the |01> state results in the expected |10> state.
 */
TEST_F(SWAPGateTest, TestState01)
{
    Complex<float> expectedState[4] = { Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |01>
    stateVector[0] = Complex<float>(0.0f, 0.0f);
    stateVector[1] = Complex<float>(1.0f, 0.0f);
    stateVector[2] = Complex<float>(0.0f, 0.0f);
    stateVector[3] = Complex<float>(0.0f, 0.0f);

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    swap(queue, stateVector_d, numQubits, 0, 1, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the SWAP gate on the |10> state.
 *
 * This test ensures that the SWAP gate applied to the |10> state results in the expected |01> state.
 */
TEST_F(SWAPGateTest, TestState10)
{
    Complex<float> expectedState[4] = { Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |10>
    stateVector[0] = Complex<float>(0.0f, 0.0f);
    stateVector[1] = Complex<float>(0.0f, 0.0f);
    stateVector[2] = Complex<float>(1.0f, 0.0f);
    stateVector[3] = Complex<float>(0.0f, 0.0f);

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    swap(queue, stateVector_d, numQubits, 0, 1, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the SWAP gate on the |11> state.
 *
 * This test ensures that the SWAP gate applied to the |11> state results in the expected |11> state.
 */
TEST_F(SWAPGateTest, TestState11)
{
    Complex<float> expectedState[4] = { Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f)};

    // State |11>
    stateVector[0] = Complex<float>(0.0f, 0.0f);
    stateVector[1] = Complex<float>(0.0f, 0.0f);
    stateVector[2] = Complex<float>(0.0f, 0.0f);
    stateVector[3] = Complex<float>(1.0f, 0.0f);

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    swap(queue, stateVector_d, numQubits, 0, 1, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

#endif