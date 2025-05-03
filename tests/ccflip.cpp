#ifndef TEST_CCZ_CPP
#define TEST_CCZ_CPP

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../headers/kernels.hpp"
#include <gtest/gtest.h>

using namespace sycl;

/**
 * @brief Testing the CCZ gate on the eight possible states: |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>.
 *
 * This test ensures that the CCZ gate gives the expected results for each of these states.
 */
class CCZGateTest : public ::testing::Test
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
        #pragma unroll
        for (size_t i = 0; i < numStates; ++i)
        {
            stateVector[i] = Complex<float>(0.0f, 0.0f);
        }
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
    const size_t numQubits = 3;              // Three qubits for the CCZ gate
    const size_t numStates = 1 << numQubits; // 8 states for 3 qubits
};

/**
 * @brief Test the CCZ gate on the |000> state.
 *
 * This test ensures that the CCZ gate applied to the |000> state results in the expected |000> state.
 */
TEST_F(CCZGateTest, TestState000)
{
    Complex<float> expectedState[8] = {Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f),
                                       Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |000>
    stateVector[0] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task); // Apply CCZ gate (control qubits 0,1 and target qubit 2)
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the CCZ gate on the |001> state.
 *
 * This test ensures that the CCZ gate applied to the |001> state results in the expected |001> state.
 */
TEST_F(CCZGateTest, TestState001)
{
    Complex<float> expectedState[8] = {Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f),
                                       Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |001>
    stateVector[1] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the CCZ gate on the |010> state.
 *
 * This test ensures that the CCZ gate applied to the |010> state results in the expected |010> state.
 */
TEST_F(CCZGateTest, TestState010)
{
    Complex<float> expectedState[8] = {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f),
                                       Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |010>
    stateVector[2] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the CCZ gate on the |011> state.
 *
 * This test ensures that the CCZ gate applied to the |011> state results in the expected |011> state.
 */
TEST_F(CCZGateTest, TestState011)
{
    Complex<float> expectedState[8] = {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f),
                                       Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |011>
    stateVector[3] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the CCZ gate on the |100> state.
 *
 * This test ensures that the CCZ gate applied to the |100> state results in the expected |100> state.
 */
TEST_F(CCZGateTest, TestState100)
{
    Complex<float> expectedState[8] = {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f),
                                       Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |100>
    stateVector[4] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the CCZ gate on the |101> state.
 *
 * This test ensures that the CCZ gate applied to the |101> state results in the expected |101> state.
 */
TEST_F(CCZGateTest, TestState101)
{
    Complex<float> expectedState[8] = {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f),
                                       Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |101>
    stateVector[5] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the CCZ gate on the |110> state.
 *
 * This test ensures that the CCZ gate applied to the |110> state results in the expected |110> state.
 */
TEST_F(CCZGateTest, TestState110)
{
    Complex<float> expectedState[8] = {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f),
                                       Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(1.0f, 0.0f), Complex<float>(0.0f, 0.0f)};

    // State |110>
    stateVector[6] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

/**
 * @brief Test the CCZ gate on the |111> state.
 *
 * This test ensures that the CCZ gate applied to the |111> state results in the expected -|111> state.
 */
TEST_F(CCZGateTest, TestState111)
{
    Complex<float> expectedState[8] = {Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f),
                                       Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(0.0f, 0.0f), Complex<float>(-1.0f, 0.0f)};

    // State |111>
    stateVector[7] = Complex<float>(1.0f, 0.0f);
    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(Complex<float>)).wait();
    controlledPhaseFlip(queue, stateVector_d, numQubits, 0, 1, 2, use_single_task);
    measure(queue, stateVector_d, numQubits, 100, use_single_task);
    queue.memcpy(stateVector, stateVector_d, numStates * sizeof(Complex<float>)).wait();

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i].real(), expectedState[i].real(), 1e-7);
        EXPECT_NEAR(stateVector[i].imag(), expectedState[i].imag(), 1e-7);
    }
}

#endif