#ifndef TEST_X_CPP
#define TEST_X_CPP

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../headers/kernels.hpp"
//#include "../headers/blochSphere.hpp"
//#include "angles.cpp"
#include <gtest/gtest.h>

using namespace sycl;

/**
 * @brief Test suite for the Y gate operation on quantum states. Uncomment bloch_sphere() to draw the first result.
 *
 * This test class verifies the behavior of the X gate on quantum states |0> and |1>.
 * Specifically, it checks that the Y gate transforms:
 * - |0> -> i|1>
 * - |1> -> -i|0>
 *
 * The tests perform state vector initialization, the application of the Z gate,
 * and verification of the results using `EXPECT_NEAR`.
 */
class YGateTest : public ::testing::Test
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
// Select the device
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
    const size_t numQubits = 1;
    const size_t numStates = 1 << numQubits; // 2
};

/**
 * @brief Test the Y gate on the |0> state.
 *
 * This test ensures that the Y gate applied to the |0> state results in the -i|1> state
 *
 */
TEST_F(YGateTest, TestState0)
{
    std::vector<Complex<float>> expectedState;
    expectedState = {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 1.0f)};
    stateVector[0] = Complex<float>(1.0f, 0.0f);
    stateVector[1] = Complex<float>(0.0f, 0.0f);

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    y(queue, stateVector_d, numQubits, 0, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    // state2angle(stateVector[0], stateVector[1], theta, phi);
    // int argc = 1;
    // char* argv[] = { (char*)"test_h_gate.cpp" };  // Mock argument
    // bloch_sphere(argc,argv);

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the Y gate on the |1> state.
 *
 * This test ensures that the Y gate applied to the |1> state results in the -i|0> state
 *
 */
TEST_F(YGateTest, TestState1)
{
    std::vector<Complex<float>> expectedState;
    expectedState = {Complex<float>(0.0f, -1.0f), Complex<float>(0.0f, 0.0f)};
    stateVector[0] = Complex<float>(0.0f, 0.0f);
    stateVector[1] = Complex<float>(1.0f, 0.0f);

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    y(queue, stateVector_d, numQubits, 0, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    // state2angle(stateVector[0], stateVector[1], theta, phi);
    // int argc = 1;
    // char* argv[] = { (char*)"test_h_gate.cpp" };  // Mock argument
    // bloch_sphere(argc,argv);

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

#endif
