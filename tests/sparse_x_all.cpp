#ifndef TEST_X_CPP
#define TEST_X_CPP

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../headers/kernels.hpp"
//#include "../headers/blochSphere.hpp"
//#include "angles.cpp"

#include <oneapi/mkl/vm.hpp>
#include <oneapi/math.hpp>
#include <oneapi/math/sparse_blas.hpp>

#if CPU
    #include <oneapi/mkl/sparse_blas.hpp>
    #include <oneapi/math/sparse_blas/detail/mklcpu/sparse_blas_ct.hpp>
#elif GPU
    #include <oneapi/math/sparse_blas/detail/cusparse/sparse_blas_ct.hpp>
#else
    #include <oneapi/math/blas/detail/generic/onemath_blas_generic.hpp>
#endif
#include <gtest/gtest.h>

using namespace sycl;

/**
 * @brief Test suite for the X gate operation on quantum states. Uncomment bloch_sphere() to draw the first result.
 *
 * This test class verifies the behavior of the X gate on quantum states |0> and |1>.
 * Specifically, it checks that the X gate transforms:
 * - |0> -> |1>
 * - |1> -> |0>
 *
 * The tests perform state vector initialization, the application of the Z gate,
 * and verification of the results using `EXPECT_NEAR`.
 */
class SparseXGateTest : public ::testing::Test
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
        stateVector = new float[numStates];
        stateVector_d = malloc_device<float>(numStates, queue);
        y = (float *)sycl::malloc_shared(numStates * sizeof(float), queue);
        std::fill(y, y + numStates, 0.0f);
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
    const size_t numQubits = 10;
    const size_t numStates = 1 << numQubits; // 2
    float alpha = 1.0f;
    float beta = 0.0f;
    float *expectedState = new float[numStates];
    float *stateVector = new float[numStates];
    float *stateVector_d;
    float *y;
};

/**
 * @brief Test the X gate on the |0000000000> state.
 *
 * This test ensures that the X gate applied to the |0000000000> state results in the |1111111111> state
 *
 */
TEST_F(SparseXGateTest, TestState0)
{
    std::fill(stateVector, stateVector + numStates, 0.0f);
    stateVector[0] = 1.0f; // |0000000000>

    std::fill(expectedState, expectedState + numStates, 0.0f);
    expectedState[numStates-1] = 1.0f; // |1111111111>

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(float)).wait();
    sparse_x_all(queue, stateVector_d, y, numQubits, alpha, beta);
    blas_measure(queue, y, numQubits, 100);
    queue.memcpy(stateVector, y, numStates * sizeof(float)).wait();

    // state2angle(stateVector[0], stateVector[1], theta, phi);
    // int argc = 1;
    // char* argv[] = { (char*)"test_h_gate.cpp" };  // Mock argument
    // bloch_sphere(argc,argv);

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i], expectedState[i], 1e-7);
    }
}

/**
 * @brief Test the X gate on the |0000000010> state.
 *
 * This test ensures that the X gate applied to the |0000000010> state results in the |1111111101> state
 *
 */
TEST_F(SparseXGateTest, TestState2)
{
    std::fill(stateVector, stateVector + numStates, 0.0f);
    stateVector[2] = 1.0f; // |0000000010>

    std::fill(expectedState, expectedState + numStates, 0.0f);
    expectedState[numStates-3] = 1.0f; // |1111111101>

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(float)).wait();
    sparse_x_all(queue, stateVector_d, y, numQubits, alpha, beta);
    blas_measure(queue, y, numQubits, 100);
    queue.memcpy(stateVector, y, numStates * sizeof(float)).wait();

    // state2angle(stateVector[0], stateVector[1], theta, phi);
    // int argc = 1;
    // char* argv[] = { (char*)"test_h_gate.cpp" };  // Mock argument
    // bloch_sphere(argc,argv);

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i], expectedState[i], 1e-7);
    }
}

/**
 * @brief Test the X gate on the |0000000101> state.
 *
 * This test ensures that the X gate applied to the |0000000101> state results in the |1111111010> state
 *
 */
TEST_F(SparseXGateTest, TestState5)
{
    std::fill(stateVector, stateVector + numStates, 0.0f);
    stateVector[5] = 1.0f; // |0000000101>

    std::fill(expectedState, expectedState + numStates, 0.0f);
    expectedState[numStates-6] = 1.0f; // |1111111010>

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(float)).wait();
    sparse_x_all(queue, stateVector_d, y, numQubits, alpha, beta);
    blas_measure(queue, y, numQubits, 100);
    queue.memcpy(stateVector, y, numStates * sizeof(float)).wait();

    // state2angle(stateVector[0], stateVector[1], theta, phi);
    // int argc = 1;
    // char* argv[] = { (char*)"test_h_gate.cpp" };  // Mock argument
    // bloch_sphere(argc,argv);

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i], expectedState[i], 1e-7);
    }
}

/**
 * @brief Test the X gate on the |1111111111> state.
 *
 * This test ensures that the X gate applied to the |1111111111> state results in the |0000000000> state
 *
 */
TEST_F(SparseXGateTest, TestState10)
{
    std::fill(stateVector, stateVector + numStates, 0.0f);
    stateVector[numStates-1] = 1.0f; // |1111111111>

    std::fill(expectedState, expectedState + numStates, 0.0f);
    expectedState[0] = 1.0f; // |0000000000>

    queue.memcpy(stateVector_d, stateVector, numStates * sizeof(float)).wait();
    sparse_x_all(queue, stateVector_d, y, numQubits, alpha, beta);
    blas_measure(queue, y, numQubits, 100);
    queue.memcpy(stateVector, y, numStates * sizeof(float)).wait();

    // state2angle(stateVector[0], stateVector[1], theta, phi);
    // int argc = 1;
    // char* argv[] = { (char*)"test_h_gate.cpp" };  // Mock argument
    // bloch_sphere(argc,argv);

    for (size_t i = 0; i < numStates; ++i)
    {
        EXPECT_NEAR(stateVector[i], expectedState[i], 1e-7);
    }
}

#endif