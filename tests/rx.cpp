#ifndef TEST_RX_CPP
#define TEST_RX_CPP

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../headers/kernels.hpp"
//#include "../headers/blochSphere.hpp"
//#include "angles.cpp"
#include <gtest/gtest.h>

using namespace sycl;

/**
 * @brief Testing the Rx gate with angles 0,pi/2,pi,-pi/2 for |0> and |1> states. Uncomment bloch_sphere() to draw the first result.
 *
 *
 */
class RxGateTest : public ::testing::Test
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
    const std::vector<float> angles = {0.0f, M_PI / 2, M_PI, -M_PI / 2};
    const float sq = std::sqrt(2.0f) / 2.0f;
};

/**
 * @brief Test the Rx gate on the |0> state.
 *
 * This test ensures that the Rx gate applied to the |0> state with angles results in the expected states
 *
 */
TEST_F(RxGateTest, TestState0_Angles)
{
    std::vector<std::vector<Complex<float>>> expectedState(4, std::vector<Complex<float>>(2));

    expectedState = {
        {Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f)},  //(1,0)
        {Complex<float>(sq, 0.0f), Complex<float>(0.0f, -sq)},     //(sqrt(2)/2, -i*sqrt(2)/2)
        {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, -1.0f)}, //(0,-i)
        {Complex<float>(sq, 0.0f), Complex<float>(0.0f, sq)}       // sqrt(2)/2, i*sqrt(2)/2
    };

    for (size_t j = 0; j < angles.size(); j++)
    {
        float angle = angles[j];
        //State |0>
        stateVector[0] = Complex<float>(1.0f, 0.0f);
        stateVector[1] = Complex<float>(0.0f, 0.0f);

        queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
        rx(queue, stateVector_d, numQubits, 0, angle, use_single_task);
        measure(queue, stateVector_d, numQubits, 100, use_single_task);
        queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

        // state2angle(stateVector[0], stateVector[1], theta, phi);
        // int argc = 1;
        // char* argv[] = { (char*)"test_rx_gate.cpp" };  // Mock argument
        // bloch_sphere(argc,argv);

        for (size_t i = 0; i < numStates; ++i)
        {
            EXPECT_NEAR(stateVector[i].real(), expectedState[j][i].real(), 1e-7);
            EXPECT_NEAR(stateVector[i].imag(), expectedState[j][i].imag(), 1e-7);
        }
    }
}

/**
 * @brief Test the Rx gate on the |1> state.
 *
 * This test ensures that the Rx gate applied to the |1> state with angles results in the expected states
 *
 */
TEST_F(RxGateTest, TestState1_Angles)
{
    std::vector<std::vector<Complex<float>>> expectedState(4, std::vector<Complex<float>>(2));

    expectedState = {
        {Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f)},  //(0,1)
        {Complex<float>(0.0f, -sq), Complex<float>(sq, 0.0f)},     //(-i*sqrt(2)/2, sqrt(2)/2)
        {Complex<float>(0.0f, -1.0f), Complex<float>(0.0f, 0.0f)}, //(-i,0)
        {Complex<float>(0.0f, sq), Complex<float>(sq, 0.0f)}       //(i*sqrt(2)/2, sqrt(2)/2)
    };

    for (size_t j = 0; j < angles.size(); j++)
    {
        float angle = angles[j];
        //State |1>
        stateVector[0] = Complex<float>(0.0f, 0.0f);
        stateVector[1] = Complex<float>(1.0f, 0.0f);

        queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
        rx(queue, stateVector_d, numQubits, 0, angle, use_single_task);
        measure(queue, stateVector_d, numQubits, 100, use_single_task);
        queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

        // state2angle(stateVector[0], stateVector[1], theta, phi);
        // int argc = 1;
        // char* argv[] = { (char*)"test_rx_gate.cpp" };  // Mock argument
        // bloch_sphere(argc,argv);

        for (size_t i = 0; i < numStates; ++i)
        {
            EXPECT_NEAR(stateVector[i].real(), expectedState[j][i].real(), 1e-7);
            EXPECT_NEAR(stateVector[i].imag(), expectedState[j][i].imag(), 1e-7);
        }
    }
}

#endif