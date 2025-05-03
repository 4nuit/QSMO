/**
 * @file kernels.cpp
 */

#include "../headers/kernels.hpp"
#include <bitset>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdlib>

#include <oneapi/mkl/vm.hpp>
#include <oneapi/math.hpp>
#include <oneapi/math/sparse_blas.hpp>
#include "../headers/example_helper.hpp"

#if CPU
#include <oneapi/mkl/sparse_blas.hpp>
#include <oneapi/math/sparse_blas/detail/mklcpu/sparse_blas_ct.hpp>
// #define ONEAPI oneapi::math::blas::mklcpu::column_major //oneapi::mkl::sparse
// #define ONEAPI_ROW oneapi::math::blas::mklcpu::row_major
#elif GPU
#include <oneapi/math/sparse_blas/detail/cusparse/sparse_blas_ct.hpp>
// #define ONEAPI oneapi::math::blas::cublas::column_major
// #define ONEAPI_ROW oneapi::math::blas::cublas::row_major
#else
#include <oneapi/math/blas/detail/generic/onemath_blas_generic.hpp>
// #define ONEAPI oneapi::math::blas::generic::column_major
// #define ONEAPI_ROW oneapi::math::blas::generic::row_major
#endif

// Forward declaration for both Circuit and BlasCircuit
std::string toBinary(int num, int numQubits);

/************************* BLASCIRCUIT CLASS ************************/

/**
 * @brief Applies the H gate (Hadamard) to a specific qubit in the state vector using BLAS gemv.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the X gate is applied.
 */
void blas_h(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target)
{
    /*
     */
}

/**
 * @brief Applies a sparse quantum gate to a state vector using BLAS kernels.
 * The gate is defined by a sizexsize matrix.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param y The device result vector that will be copied on stateVector with SpMV (do init to a null vector).
 * @param ia The device array containing the indexes of the lines of non zero elements in the computed gate
 * @param ja The device array containing the indexes of the columns of non zero elements in the computed gate
 * @param A The device array containing the gate in the COO format
 * @param size The number of columns of the gate
 * @param nnz The number of non zero elements of the gate.
 * @param alpha The alpha parameter in SpMV (do init to 1)
 * @param beta The beta parameter in SpmV (do init to 0)
 * @param target The target qubit on which the gate is applied.
 */
void apply_spmv_gate(sycl::queue &queue, float *stateVector_d, int numStates, float *y, int *ia, int *ja, float *A, int size, int nnz, float alpha, float beta, int target)
{

#if CPU
    oneapi::math::backend_selector<oneapi::math::backend::mklcpu> selector{queue};
#elif GPU
    oneapi::math::backend_selector<oneapi::math::backend::cusparse> selector{queue};
//  Wont work, but allows compilation for the previous implem
#elif FPGA_SIMULATOR
    sycl::ext::intel::fpga_simulator_selector selector;
#elif FPGA_HARDWARE
    sycl::ext::intel::fpga_selector selector;
#elif FPGA_EMULATOR
    sycl::ext::intel::fpga_emulator_selector selector;
#endif

    oneapi::math::transpose transA = oneapi::math::transpose::nontrans;
    oneapi::math::sparse::spmv_alg alg = oneapi::math::sparse::spmv_alg::default_alg; // or coo_alg1
    oneapi::math::sparse::matrix_view A_view;

    // Create and initixlize handle for a Sparse Matrix in COO format sorted by rows
    oneapi::math::sparse::matrix_handle_t A_handle = nullptr;
    oneapi::math::sparse::init_coo_matrix(selector, &A_handle, size, size, nnz, oneapi::math::index_base::zero, ia, ja, A);
    oneapi::math::sparse::set_matrix_property(selector, A_handle, oneapi::math::sparse::matrix_property::sorted_by_rows); // symmetric not available for cuBLAS

    oneapi::math::sparse::dense_vector_handle_t x_handle = nullptr;
    oneapi::math::sparse::dense_vector_handle_t y_handle = nullptr;
    oneapi::math::sparse::init_dense_vector(selector, &x_handle, size, stateVector_d);
    oneapi::math::sparse::init_dense_vector(selector, &y_handle, size, y);

    oneapi::math::sparse::spmv_descr_t descr = nullptr;
    oneapi::math::sparse::init_spmv_descr(selector, &descr);

    std::size_t workspace_size = 0;
    oneapi::math::sparse::spmv_buffer_size(selector, transA, &alpha, A_view, A_handle, x_handle, &beta, y_handle, alg, descr, workspace_size);
    void *workspace = sycl::malloc_device(workspace_size, queue);

    // Optimize & Run SpMV
    auto ev_opt = oneapi::math::sparse::spmv_optimize(selector, transA, &alpha, A_view, A_handle, x_handle, &beta, y_handle, alg, descr, workspace);
    // auto ev_spmv = oneapi::math::sparse::spmv(selector, transA, &alpha, A_view, A_handle, x_handle, &beta, y_handle, alg, descr, {ev_opt});

    // Start event for SpMV
    sycl::event start_event = queue.submit([&](sycl::handler &cgh)
                                           {
     cgh.depends_on(ev_opt);  // Ensure ev_opt completes before spmv
     cgh.single_task<class spmv_start_event>([=]() {
         // Do nothing in the kernel, it's just to capture time
     }); });

    // Run SpMV
    auto ev_spmv = oneapi::math::sparse::spmv(selector, transA, &alpha, A_view, A_handle, x_handle,
                                              &beta, y_handle, alg, descr, {start_event});

    // End event after SpMV
    sycl::event end_event = queue.submit([&](sycl::handler &cgh)
                                         {
     cgh.depends_on(ev_spmv);  // Ensure spmv completes before measuring end time
     cgh.single_task<class spmv_end_event>([=]() {
         // Do nothing, just capture time
     }); });

    // Wait for the end event to finish and capture elapsed time
    end_event.wait_and_throw();
    auto start_time = start_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time = end_event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto spmv_time_ms = (end_time - start_time) / 1.0e9; // Convert from nanoseconds to milliseconds

    std::cout << "SPMV execution time: " << spmv_time_ms << " s" << std::endl;

    std::vector<sycl::event> release_events;
    release_events.push_back(
        oneapi::math::sparse::release_dense_vector(selector, x_handle, {ev_spmv}));
    release_events.push_back(
        oneapi::math::sparse::release_dense_vector(selector, y_handle, {ev_spmv}));
    release_events.push_back(
        oneapi::math::sparse::release_sparse_matrix(selector, A_handle, {ev_spmv}));
    release_events.push_back(
        oneapi::math::sparse::release_spmv_descr(selector, descr, {ev_spmv}));
    for (auto event : release_events)
    {
        event.wait_and_throw();
    }

    // queue.memcpy(stateVector_d, y, numStates * sizeof(float));
}

/**
 * @brief Applies the X gate (NOT) to a specific qubit in the state vector using Sparse representation.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the X gate is applied.
 */
void sparse_x(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target)
{
    /*
     */
}

/**
 * @brief Applies the X gate (NOT) to ALL qubits in the state vector using Sparse representation.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 */
void sparse_x_all(sycl::queue &queue, float *stateVector_d, float *y, const unsigned int numQubits, float alpha, float beta)
{
    int numStates = 1 << numQubits; // floats
    // size = numQubits = nnz for X GATE => diagonal elements ( nnz >= size in general).
    int nnz = numStates;
    int size = numStates;

    // Setting Non Zero elements for X gate
    int host_ix[numStates];
    int host_jx[numStates];
    for (int i = 0; i < nnz; i++)
    {
        host_ix[i] = i;
        host_jx[i] = nnz - (i + 1);
    }

    int *ix = malloc_device<int>(nnz, queue);
    int *jx = malloc_device<int>(nnz, queue);
    float *x = new float[nnz];
    float *X = malloc_device<float>(nnz, queue);

    queue.memcpy(ix, host_ix, nnz * sizeof(int)).wait_and_throw();
    queue.memcpy(jx, host_jx, nnz * sizeof(int)).wait_and_throw();
    // Setting Non Zero values for X gate elements
    std::fill(x, x + nnz, 1.0f);
    queue.memcpy(X, x, nnz * sizeof(float)).wait();

    int target = 0; // non used here
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

    apply_spmv_gate(queue, stateVector_d, numStates, y, ix, jx, X, size, nnz, alpha, beta, target);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Apply SpMV (std::chrono) elapsed time: " << duration.count() << "ms\n";
}

/**
 * @brief Applies the Z gate (phase flip) to a specific qubit in the state vector using Sparse representation.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the Z gate is applied.
 */
void sparse_z(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target)
{
    /*
     */
}

/**
 * @brief Applies the Toffoli gate (CCNOT) to ALL qubits in the state vector using Sparse representation.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 */
void sparse_ccnot_all(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta)
{
    int numStates = 1 << numQubits; // floats
    // size = numQubits = nnz for X GATE => diagonal elements ( nnz >= size in general).
    int nnz = numStates;
    int size = numStates;

    // Setting Non Zero elements for X gate
    int host_ic[nnz];
    int host_jc[nnz];
    for (int i = 0; i < nnz - 2; i++)
    {
        host_ic[i] = i;
        host_jc[i] = i;
    }

    host_ic[nnz - 2] = nnz - 1;
    host_jc[nnz - 2] = nnz - 2;
    host_ic[nnz - 1] = nnz - 2;
    host_jc[nnz - 1] = nnz - 1;

    // NEED TO FIX USING MALLOC DEVICE CF SPARSE_X_ALL
    int *ic = (int *)sycl::malloc_shared(nnz * sizeof(int), queue);
    int *jc = (int *)sycl::malloc_shared(nnz * sizeof(int), queue);
    float *CCNOT = (float *)sycl::malloc_shared(nnz * sizeof(float), queue);

    queue.memcpy(ic, host_ic, nnz * sizeof(int)).wait_and_throw();
    queue.memcpy(jc, host_jc, nnz * sizeof(int)).wait_and_throw();
    // Setting Non Zero values for X gate elements
    for (int i = 0; i < nnz - 2; i++)
    {
        CCNOT[i] = 1.0f;
    }
    CCNOT[nnz - 1] = 0;

    float *y = (float *)sycl::malloc_shared(numStates * sizeof(float), queue);
    for (int i = 0; i < numStates; i++)
    {
        y[i] = 0.0f;
    }

    int target = 0; // non used here
    apply_spmv_gate(queue, stateVector_d, numStates, y, ic, jc, CCNOT, size, nnz, alpha, beta, target);
}

/**
 * @brief Applies the controlledPhaseFlip (CCZ) using Sparse representation gate to 3 qubits in the state vector.
 * The Toffoli gate applies a NOT on the target qubit if both control qubits are |1>.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target1 The first control qubit on which the Toffoli gate is applied.
 * @param target2 The second control qubit on which the Toffoli gate is applied.
 * @param target3 The target qubit on which the Toffoli gate is applied.
 */
void sparse_ccflip(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target1, int target2, int target3)
{
    /*
     */
}

/**
 * @brief Simulates a quantum measurement by randomly sampling from the state
 * vector according to the probabilities of the states.
 * It prints the resulting measurement outcomes.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param samples The number of samples to draw from the probability distribution.
 */
void blas_measure(sycl::queue &queue, float *stateVector_d, int numQubits, int samples)
{
    const int numStates = std::pow(2, numQubits);
    std::vector<float> probs(numStates);
    float *probs_d = malloc_device<float>(numStates, queue);

    queue.parallel_for<class Sparse_Proba>(sycl::range<1>(numStates), [=](sycl::item<1> item)
                                           {
            int global_id = item.get_id(0);
            float amp = stateVector_d[global_id];
            probs_d[global_id] = std::abs(amp * amp); })
        .wait();

    auto event = queue.memcpy(probs.data(), probs_d, numStates * sizeof(float));
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Measure elapsed time: " << (end - start) / 1.0e9 << "seconds\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.data(), probs.data() + numStates);
    std::vector<int> result(numStates);

    for (int i = 0; i < samples; ++i)
    {
        int idx = dist(gen);
        ++result[idx];
    }

    for (int i = 0; i < numStates; ++i)
    {
        if (result[i] != 0)
        {
            std::cout << "State " << i << " | " << toBinary(i, numQubits) << "> realized with proba: " << result[i] << "%" << std::endl;
        }
    }
    sycl::free(probs_d, queue);
    probs_d = nullptr;
}

/************************* CIRCUIT CLASS ************************/

/**
 * @brief Converts an integer into its binary representation as a string,
 * with the binary value padded to represent `numQubits` bits.
 *
 * @param num The integer to convert.
 * @param numQubits The number of qubits (bits) to pad the binary string to.
 * @return The binary string representation of the number, padded to `numQubits` bits.
 */
std::string toBinary(int num, int numQubits)
{
    std::string result;
    for (int i = numQubits - 1; i >= 0; i--)
    {
        int bit = (num >> i) & 1;
        result += std::to_string(bit);
    }
    return result;
}

/**
 * @brief Removes the target-th bit from the binary representation of a number.
 * This is effectively setting the target bit to 0,
 * and shifting the remaining bits accordingly.
 *
 * @param n The integer to modify.
 * @param target The position of the bit to clear (0-indexed).
 * @return The modified integer with the target-th bit cleared.
 */
int nth_cleared(int n, int target)
{
    int mask = (1 << target) - 1; // 00010000 (1 in nth=target position)
    int not_mask = ~mask;         // 11101111
    return (n & mask) | ((n & not_mask) << 1);
}

/**
 * @brief Applies a 1-qubit quantum gate to a state vector using parallel_for kernels.
 * The gate is defined by a 2x2 matrix, where the values A, B, C, and D
 * represent the elements of the matrix that operate on the quantum state.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param target The target qubit on which the gate is applied.
 * @param A The matrix element that multiplies the |0⟩ state amplitude in the 1st row.
 * @param B The matrix element that multiplies the |1⟩ state amplitude in the 1st row.
 * @param C The matrix element that multiplies the |0⟩ state amplitude in the 2nd row.
 * @param D The matrix element that multiplies the |1⟩ state amplitude in the 2nd row.
 */
void apply_gate(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                const int target,
                const Complex<float> A,
                const Complex<float> B,
                const Complex<float> C,
                const Complex<float> D)
{
    auto event = queue.parallel_for<class Gate1>(sycl::range<1>(numStates), [=](sycl::item<1> item)
                                                 {
                int global_id = item.get_id(0);
                const int zero_state = nth_cleared(global_id,target);
                const int one_state = zero_state | (1 << target);
                Complex<float> zero_amp = stateVector_d[zero_state];
                Complex<float> one_amp = stateVector_d[one_state];
                stateVector_d[zero_state] = A * zero_amp + B * one_amp;
                stateVector_d[one_state] =  C * zero_amp + D * one_amp; });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Apply gate1_parallel elapsed time: " << (end - start) / 1.0e9 << "seconds\n";
}

/**
 * @brief Applies a 1-qubit quantum gate to a state vector using single_task kernels.
 * The gate is defined by a 2x2 matrix, where the values A, B, C, and D
 * represent the elements of the matrix that operate on the quantum state.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param target The target qubit on which the gate is applied.
 * @param A The matrix element that multiplies the |0⟩ state amplitude in the 1st row.
 * @param B The matrix element that multiplies the |1⟩ state amplitude in the 1st row.
 * @param C The matrix element that multiplies the |0⟩ state amplitude in the 2nd row.
 * @param D The matrix element that multiplies the |1⟩ state amplitude in the 2nd row.
 */
void apply_gate_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                       const int target,
                       const Complex<float> A,
                       const Complex<float> B,
                       const Complex<float> C,
                       const Complex<float> D)
{
    auto event = queue.single_task<class Gate1_single>([=]()
                                                       {
            int global_id;
#pragma unroll 1
            for (global_id = 0; global_id < numStates; ++global_id)
            {
                const int zero_state = nth_cleared(global_id, target);
                const int one_state = zero_state | (1 << target);
                Complex<float> zero_amp = stateVector_d[zero_state];
                Complex<float> one_amp = stateVector_d[one_state];
                stateVector_d[zero_state] = A * zero_amp + B * one_amp;
                stateVector_d[one_state] = C * zero_amp + D * one_amp;
            } });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Apply gate1_single elapsed time: " << (end - start) / 1.0e9 << "seconds\n";
}

/**
 * @brief Applies a 2-qubit quantum gate to a state vector with parallel_for kernels.
 * The gate is defined by a 4x4 matrix, where the values A, B, C, ..., P
 * represent the elements of the matrix that operate on the quantum state.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param target The target qubit on which the gate is applied.
 * @param A The matrix element that multiplies the |00⟩ state amplitude in the 1st row.
 * @param B The matrix element that multiplies the |01⟩ state amplitude in the 1st row.
 * @param C The matrix element that multiplies the |10⟩ state amplitude in the 1st row.
 * @param D The matrix element that multiplies the |11⟩ state amplitude in the 1st row.
 * @param E The matrix element that multiplies the |00⟩ state amplitude in the 2nd row.
 * @param F The matrix element that multiplies the |01⟩ state amplitude in the 2nd row.
 * @param G The matrix element that multiplies the |10⟩ state amplitude in the 2nd row.
 * @param H The matrix element that multiplies the |11⟩ state amplitude in the 2nd row.
 * @param I The matrix element that multiplies the |00⟩ state amplitude in the 3rd row.
 * @param J The matrix element that multiplies the |01⟩ state amplitude in the 3rd row.
 * @param K The matrix element that multiplies the |10⟩ state amplitude in the 3rd row.
 * @param L The matrix element that multiplies the |11⟩ state amplitude in the 3rd row.
 * @param M The matrix element that multiplies the |00⟩ state amplitude in the 4th row.
 * @param N The matrix element that multiplies the |01⟩ state amplitude in the 4th row.
 * @param O The matrix element that multiplies the |10⟩ state amplitude in the 4th row.
 * @param P The matrix element that multiplies the |11⟩ state amplitude in the 4th row.
 */
void apply_gate2(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                 const int target1, const int target2,
                 const Complex<float> A, const Complex<float> B, const Complex<float> C, const Complex<float> D,
                 const Complex<float> E, const Complex<float> F, const Complex<float> G, const Complex<float> H,
                 const Complex<float> I, const Complex<float> J, const Complex<float> K, const Complex<float> L,
                 const Complex<float> M, const Complex<float> N, const Complex<float> O, const Complex<float> P)

{
    auto event = queue.parallel_for<class Gate2>(sycl::range<1>(numStates), [=](sycl::item<1> item)
                                                 {
            int global_id = item.get_id(0);
            int state_00 = nth_cleared(global_id, target1) & nth_cleared(global_id, target2);
            int state_01 = state_00 | (1 << target1);
            int state_10 = state_00 | (1 << target2);
            int state_11 = state_01 | (1 << target2);
    
            Complex<float> amp_00 = stateVector_d[state_00];
            Complex<float> amp_01 = stateVector_d[state_01];
            Complex<float> amp_10 = stateVector_d[state_10];
            Complex<float> amp_11 = stateVector_d[state_11];
    
            stateVector_d[state_00] = A * amp_00 + B * amp_01 + C * amp_10 + D * amp_11;
            stateVector_d[state_01] = E * amp_00 + F * amp_01 + G * amp_10 + H * amp_11;
            stateVector_d[state_10] = I * amp_00 + J * amp_01 + K * amp_10 + L * amp_11;
            stateVector_d[state_11] = M * amp_00 + N * amp_01 + O * amp_10 + P * amp_11; });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Apply gate2_parallel elapsed time: " << (end - start) / 1.0e9 << " seconds\n";
}

/**
 * @brief Applies a 2-qubit quantum gate to a state vector with single_task kernels.
 * The gate is defined by a 4x4 matrix, where the values A, B, C, ..., P
 * represent the elements of the matrix that operate on the quantum state.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param target The target qubit on which the gate is applied.
 * @param A The matrix element that multiplies the |00⟩ state amplitude in the 1st row.
 * @param B The matrix element that multiplies the |01⟩ state amplitude in the 1st row.
 * @param C The matrix element that multiplies the |10⟩ state amplitude in the 1st row.
 * @param D The matrix element that multiplies the |11⟩ state amplitude in the 1st row.
 * @param E The matrix element that multiplies the |00⟩ state amplitude in the 2nd row.
 * @param F The matrix element that multiplies the |01⟩ state amplitude in the 2nd row.
 * @param G The matrix element that multiplies the |10⟩ state amplitude in the 2nd row.
 * @param H The matrix element that multiplies the |11⟩ state amplitude in the 2nd row.
 * @param I The matrix element that multiplies the |00⟩ state amplitude in the 3rd row.
 * @param J The matrix element that multiplies the |01⟩ state amplitude in the 3rd row.
 * @param K The matrix element that multiplies the |10⟩ state amplitude in the 3rd row.
 * @param L The matrix element that multiplies the |11⟩ state amplitude in the 3rd row.
 * @param M The matrix element that multiplies the |00⟩ state amplitude in the 4th row.
 * @param N The matrix element that multiplies the |01⟩ state amplitude in the 4th row.
 * @param O The matrix element that multiplies the |10⟩ state amplitude in the 4th row.
 * @param P The matrix element that multiplies the |11⟩ state amplitude in the 4th row.
 */
void apply_gate2_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                        const int target1, const int target2,
                        const Complex<float> A, const Complex<float> B, const Complex<float> C, const Complex<float> D,
                        const Complex<float> E, const Complex<float> F, const Complex<float> G, const Complex<float> H,
                        const Complex<float> I, const Complex<float> J, const Complex<float> K, const Complex<float> L,
                        const Complex<float> M, const Complex<float> N, const Complex<float> O, const Complex<float> P)

{
    auto event = queue.single_task<class Gate2_single>([=]()
                                                       {
            int global_id;
#pragma unroll 1
            for (global_id = 0; global_id < numStates; ++global_id)
            {
                int state_00 = nth_cleared(global_id, target1) & nth_cleared(global_id, target2);
                int state_01 = state_00 | (1 << target1);
                int state_10 = state_00 | (1 << target2);
                int state_11 = state_01 | (1 << target2);
    
                Complex<float> amp_00 = stateVector_d[state_00];
                Complex<float> amp_01 = stateVector_d[state_01];
                Complex<float> amp_10 = stateVector_d[state_10];
                Complex<float> amp_11 = stateVector_d[state_11];
    
                stateVector_d[state_00] = A * amp_00 + B * amp_01 + C * amp_10 + D * amp_11;
                stateVector_d[state_01] = E * amp_00 + F * amp_01 + G * amp_10 + H * amp_11;
                stateVector_d[state_10] = I * amp_00 + J * amp_01 + K * amp_10 + L * amp_11;
                stateVector_d[state_11] = M * amp_00 + N * amp_01 + O * amp_10 + P * amp_11; 
            } });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Apply gate2_single elapsed time: " << (end - start) / 1.0e9 << " seconds\n";
}

/**
 * @brief Applies a 3-qubit quantum gate to a state vector with parallel_for kernels.
 * The gate is defined by a 8x8 matrix, where the values AA, AB, AC, ... AZ, BA, ... BZ, CA, ... CL
 * represent the elements of the matrix that operate on the quantum state.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param target The target qubit on which the gate is applied.
 * @param A The matrix element that multiplies the |000⟩ state amplitude in the 1st row (/8).
 */
void apply_gate3(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                 const int target1, const int target2, const int target3,
                 const Complex<float> AA, const Complex<float> AB, const Complex<float> AC, const Complex<float> AD, const Complex<float> AE, const Complex<float> AF, const Complex<float> AG, const Complex<float> AH,
                 const Complex<float> BA, const Complex<float> BB, const Complex<float> BC, const Complex<float> BD, const Complex<float> BE, const Complex<float> BF, const Complex<float> BG, const Complex<float> BH,
                 const Complex<float> CA, const Complex<float> CB, const Complex<float> CC, const Complex<float> CD, const Complex<float> CE, const Complex<float> CF, const Complex<float> CG, const Complex<float> CH,
                 const Complex<float> DA, const Complex<float> DB, const Complex<float> DC, const Complex<float> DD, const Complex<float> DE, const Complex<float> DF, const Complex<float> DG, const Complex<float> DH,
                 const Complex<float> EA, const Complex<float> EB, const Complex<float> EC, const Complex<float> ED, const Complex<float> EE, const Complex<float> EF, const Complex<float> EG, const Complex<float> EH,
                 const Complex<float> FA, const Complex<float> FB, const Complex<float> FC, const Complex<float> FD, const Complex<float> FE, const Complex<float> FF, const Complex<float> FG, const Complex<float> FH,
                 const Complex<float> GA, const Complex<float> GB, const Complex<float> GC, const Complex<float> GD, const Complex<float> GE, const Complex<float> GF, const Complex<float> GG, const Complex<float> GH,
                 const Complex<float> HA, const Complex<float> HB, const Complex<float> HC, const Complex<float> HD, const Complex<float> HE, const Complex<float> HF, const Complex<float> HG, const Complex<float> HH)
{
    auto event = queue.parallel_for<class Gate3>(sycl::range<1>(numStates), [=](sycl::item<1> item)
                                                 {
            int global_id = item.get_id(0);
            int state_000 = nth_cleared(global_id, target1) & nth_cleared(global_id, target2) & nth_cleared(global_id, target3);
            int state_001 = state_000 | (1 << target1);  // Set the 1st qubit to 1
            int state_010 = state_000 | (1 << target2);  // Set the 2nd qubit to 1
            int state_011 = state_001 | (1 << target2);  // Set both the 1st and 2nd qubits to 1
            int state_100 = state_000 | (1 << target3);  // Set the 3rd qubit to 1
            int state_101 = state_001 | (1 << target3);  // Set the 1st and 3rd qubits to 1
            int state_110 = state_010 | (1 << target3);  // Set the 2nd and 3rd qubits to 1
            int state_111 = state_011 | (1 << target3);  // Set all 3 qubits to 1
    
            Complex<float> amp_000 = stateVector_d[state_000];
            Complex<float> amp_001 = stateVector_d[state_001];
            Complex<float> amp_010 = stateVector_d[state_010];
            Complex<float> amp_011 = stateVector_d[state_011];
            Complex<float> amp_100 = stateVector_d[state_100];
            Complex<float> amp_101 = stateVector_d[state_101];
            Complex<float> amp_110 = stateVector_d[state_110];
            Complex<float> amp_111 = stateVector_d[state_111];
    
            stateVector_d[state_000] = AA * amp_000 + AB * amp_001 + AC * amp_010 + AD * amp_011 + AE * amp_100 + AF * amp_101 + AG * amp_110 + AH * amp_111;
            stateVector_d[state_001] = BA * amp_000 + BB * amp_001 + BC * amp_010 + BD * amp_011 + BE * amp_100 + BF * amp_101 + BG * amp_110 + BH * amp_111;
            stateVector_d[state_010] = CA * amp_000 + CB * amp_001 + CC * amp_010 + CD * amp_011 + CE * amp_100 + CF * amp_101 + CG * amp_110 + CH * amp_111;
            stateVector_d[state_011] = DA * amp_000 + DB * amp_001 + DC * amp_010 + DD * amp_011 + DE * amp_100 + DF * amp_101 + DG * amp_110 + DH * amp_111;
            stateVector_d[state_100] = EA * amp_000 + EB * amp_001 + EC * amp_010 + ED * amp_011 + EE * amp_100 + EF * amp_101 + EG * amp_110 + EH * amp_111;
            stateVector_d[state_101] = FA * amp_000 + FB * amp_001 + FC * amp_010 + FD * amp_011 + FE * amp_100 + FF * amp_101 + FG * amp_110 + FH * amp_111;
            stateVector_d[state_110] = GA * amp_000 + GB * amp_001 + GC * amp_010 + GD * amp_011 + GE * amp_100 + GF * amp_101 + GG * amp_110 + GH * amp_111;
            stateVector_d[state_111] = HA * amp_000 + HB * amp_001 + HC * amp_010 + HD * amp_011 + HE * amp_100 + HF * amp_101 + HG * amp_110 + HH * amp_111; });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Apply gate3_parallel elapsed time: " << (end - start) / 1.0e9 << " seconds\n";
}

/**
 * @brief Applies a 3-qubit quantum gate to a state vector with single_task kernels.
 * The gate is defined by a 8x8 matrix, where the values AA, AB, AC, ... AZ, BA, ... BZ, CA, ... CL
 * represent the elements of the matrix that operate on the quantum state.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param target The target qubit on which the gate is applied.
 * @param A The matrix element that multiplies the |000⟩ state amplitude in the 1st row (/8).
 */
void apply_gate3_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                        const int target1, const int target2, const int target3,
                        const Complex<float> AA, const Complex<float> AB, const Complex<float> AC, const Complex<float> AD, const Complex<float> AE, const Complex<float> AF, const Complex<float> AG, const Complex<float> AH,
                        const Complex<float> BA, const Complex<float> BB, const Complex<float> BC, const Complex<float> BD, const Complex<float> BE, const Complex<float> BF, const Complex<float> BG, const Complex<float> BH,
                        const Complex<float> CA, const Complex<float> CB, const Complex<float> CC, const Complex<float> CD, const Complex<float> CE, const Complex<float> CF, const Complex<float> CG, const Complex<float> CH,
                        const Complex<float> DA, const Complex<float> DB, const Complex<float> DC, const Complex<float> DD, const Complex<float> DE, const Complex<float> DF, const Complex<float> DG, const Complex<float> DH,
                        const Complex<float> EA, const Complex<float> EB, const Complex<float> EC, const Complex<float> ED, const Complex<float> EE, const Complex<float> EF, const Complex<float> EG, const Complex<float> EH,
                        const Complex<float> FA, const Complex<float> FB, const Complex<float> FC, const Complex<float> FD, const Complex<float> FE, const Complex<float> FF, const Complex<float> FG, const Complex<float> FH,
                        const Complex<float> GA, const Complex<float> GB, const Complex<float> GC, const Complex<float> GD, const Complex<float> GE, const Complex<float> GF, const Complex<float> GG, const Complex<float> GH,
                        const Complex<float> HA, const Complex<float> HB, const Complex<float> HC, const Complex<float> HD, const Complex<float> HE, const Complex<float> HF, const Complex<float> HG, const Complex<float> HH)
{
    auto event = queue.single_task<class Gate3_single>([=]()
                                                       {
            int global_id;
#pragma unroll 1
            for(global_id = 0; global_id < numStates; ++global_id)
            {
                int state_000 = nth_cleared(global_id, target1) & nth_cleared(global_id, target2) & nth_cleared(global_id, target3);
                int state_001 = state_000 | (1 << target1);  // Set the 1st qubit to 1
                int state_010 = state_000 | (1 << target2);  // Set the 2nd qubit to 1
                int state_011 = state_001 | (1 << target2);  // Set both the 1st and 2nd qubits to 1
                int state_100 = state_000 | (1 << target3);  // Set the 3rd qubit to 1
                int state_101 = state_001 | (1 << target3);  // Set the 1st and 3rd qubits to 1
                int state_110 = state_010 | (1 << target3);  // Set the 2nd and 3rd qubits to 1
                int state_111 = state_011 | (1 << target3);  // Set all 3 qubits to 1
    
                Complex<float> amp_000 = stateVector_d[state_000];
                Complex<float> amp_001 = stateVector_d[state_001];
                Complex<float> amp_010 = stateVector_d[state_010];
                Complex<float> amp_011 = stateVector_d[state_011];
                Complex<float> amp_100 = stateVector_d[state_100];
                Complex<float> amp_101 = stateVector_d[state_101];
                Complex<float> amp_110 = stateVector_d[state_110];
                Complex<float> amp_111 = stateVector_d[state_111];
    
                stateVector_d[state_000] = AA * amp_000 + AB * amp_001 + AC * amp_010 + AD * amp_011 + AE * amp_100 + AF * amp_101 + AG * amp_110 + AH * amp_111;
                stateVector_d[state_001] = BA * amp_000 + BB * amp_001 + BC * amp_010 + BD * amp_011 + BE * amp_100 + BF * amp_101 + BG * amp_110 + BH * amp_111;
                stateVector_d[state_010] = CA * amp_000 + CB * amp_001 + CC * amp_010 + CD * amp_011 + CE * amp_100 + CF * amp_101 + CG * amp_110 + CH * amp_111;
                stateVector_d[state_011] = DA * amp_000 + DB * amp_001 + DC * amp_010 + DD * amp_011 + DE * amp_100 + DF * amp_101 + DG * amp_110 + DH * amp_111;
                stateVector_d[state_100] = EA * amp_000 + EB * amp_001 + EC * amp_010 + ED * amp_011 + EE * amp_100 + EF * amp_101 + EG * amp_110 + EH * amp_111;
                stateVector_d[state_101] = FA * amp_000 + FB * amp_001 + FC * amp_010 + FD * amp_011 + FE * amp_100 + FF * amp_101 + FG * amp_110 + FH * amp_111;
                stateVector_d[state_110] = GA * amp_000 + GB * amp_001 + GC * amp_010 + GD * amp_011 + GE * amp_100 + GF * amp_101 + GG * amp_110 + GH * amp_111;
                stateVector_d[state_111] = HA * amp_000 + HB * amp_001 + HC * amp_010 + HD * amp_011 + HE * amp_100 + HF * amp_101 + HG * amp_110 + HH * amp_111; 
            } });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Apply gate3_single elapsed time: " << (end - start) / 1.0e9 << " seconds\n";
}

/**
 * @brief Computes the probability of each state in the quantum state vector
 * by calculating the magnitude squared of each amplitude using a parallel_for kernel.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param probs_d The device memory to store the calculated probabilities.
 */
void get_proba(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates, float *probs_d)
{
    queue.parallel_for<class Proba>(sycl::range<1>(numStates), [=](sycl::item<1> item)
                                    {
            int global_id = item.get_id(0);
            Complex<float> amp = stateVector_d[global_id];
            probs_d[global_id] = Absol_v(amp * amp); })
        .wait();
}

/**
 * @brief Computes the probability of each state in the quantum state vector
 * by calculating the magnitude squared of each amplitude using a single_task kernel.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the state vector.
 * @param probs_d The device memory to store the calculated probabilities.
 */
void get_proba_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates, float *probs_d)
{
    queue.single_task<class Proba_single>([=]()
                                          {
            int global_id;
#pragma unroll 1
            for(global_id = 0; global_id < numStates; ++global_id){
                Complex<float> amp = stateVector_d[global_id];
                probs_d[global_id] = Absol_v(amp * amp); 
            } })
        .wait();
}

/**
 * @brief Simulates a quantum measurement by randomly sampling from the state
 * vector according to the probabilities of the states.
 * It prints the resulting measurement outcomes.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param samples The number of samples to draw from the probability distribution.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void measure(sycl::queue &queue, Complex<float> *stateVector_d, int numQubits, int samples, bool use_single_task)
{
    const int numStates = std::pow(2, numQubits);
    std::vector<float> probs(numStates);
    float *probs_d = malloc_device<float>(numStates, queue);

    if (use_single_task)
    {
        get_proba_single(queue, stateVector_d, numStates, probs_d);
    }
    else
    {
        get_proba(queue, stateVector_d, numStates, probs_d);
    }
    // oneapi::mkl::vm::sqr(queue, numStates, stateVector_d, probs_d);

    auto event = queue.memcpy(probs.data(), probs_d, numStates * sizeof(float));
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Measure elapsed time: " << (end - start) / 1.0e9 << "seconds\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.data(), probs.data() + numStates);
    std::vector<int> result(numStates);

    for (int i = 0; i < samples; ++i)
    {
        int idx = dist(gen);
        ++result[idx];
    }

    for (int i = 0; i < numStates; ++i)
    {
        if (result[i] != 0)
        {
            std::cout << "State " << i << " | " << toBinary(i, numQubits) << "> realized with proba: " << result[i] << "%" << std::endl;
        }
    }
    sycl::free(probs_d, queue);
    probs_d = nullptr;
}

/**
 * @brief Applies the Hadamard gate (H) to a specific qubit in the state vector using parallel_for.
 * The Hadamard gate creates an equal superposition of |0⟩ and |1⟩ for the target qubit.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the Hadamard gate is applied.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void h(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target, bool use_single_task)
{
    float sqrt_2 = std::sqrt(2);
    Complex<float> A(1.0 / sqrt_2, 0.0);
    Complex<float> B = A;
    Complex<float> C = A;
    Complex<float> D = -A;

    if (use_single_task)
    {
        apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
    else
    {
        apply_gate(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
}

/**
 * @brief Applies the Hadamard gate (H) to numStates qubit in the state vector using parallel_for kernel.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the quantum state.
 */
void h_n(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates)
{
    if (numStates == 1)
    {
        return;
    }

    h_n(queue, stateVector_d, numStates / 2);
    h_n(queue, stateVector_d + numStates / 2, numStates / 2);

    auto event = queue.parallel_for<class Hn>(sycl::range<1>(numStates / 2), [=](sycl::item<1> item)
                                              {
            int global_id = item.get_id(0);
            Complex<float> tmp = stateVector_d[global_id];
            stateVector_d[global_id] = tmp + stateVector_d[numStates/2 + global_id];
            stateVector_d[numStates/2 + global_id] = tmp - stateVector_d[numStates/2 + global_id]; });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Hn_parallel elapsed time (measure): " << (end - start) / 1.0e9 << "seconds\n";
}

/**
 * @brief Applies the Hadamard gate (H) to numStates qubit in the state vector using single_task kernel.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numStates The number of states in the quantum state.
 */
void h_n_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates)
{
    if (numStates == 1)
    {
        return;
    }
    h_n(queue, stateVector_d, numStates / 2);
    h_n(queue, stateVector_d + numStates / 2, numStates / 2);

    auto event = queue.single_task<class Hn_single>([=]()
                                                    {
            int global_id;
#pragma unroll 1
            for(global_id = 0; global_id < numStates/2; ++global_id)
            {
                Complex<float> tmp = stateVector_d[global_id];
                stateVector_d[global_id] = tmp + stateVector_d[numStates/2 + global_id];
                stateVector_d[numStates/2 + global_id] = tmp - stateVector_d[numStates/2 + global_id]; 
            } });
    event.wait();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Hn_single elapsed time (measure): " << (end - start) / 1.0e9 << "seconds\n";
}

/**
 * @brief Applies the X gate (NOT) to a specific qubit in the state vector.
 * The X gate changes the state |0> to |1> and vice-versa.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the X gate is applied.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void x(sycl::queue &queue, Complex<float> *stateVector_d,
       const unsigned int numQubits,
       const int target, bool use_single_task)
{
    Complex<float> A(0.0f, 0.0f);
    Complex<float> B(1.0f, 0.0f);
    Complex<float> C = B;
    Complex<float> D = A;

    if (use_single_task)
    {
        apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
    else
    {
        apply_gate(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
}

/**
 * @brief Applies the Y gate to a specific qubit in the state vector.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the Y gate is applied.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void y(sycl::queue &queue, Complex<float> *stateVector_d,
       const unsigned int numQubits,
       const int target, bool use_single_task)
{
    Complex<float> A(0.0f, 0.0f);
    Complex<float> B(0.0f, -1.0f);
    Complex<float> C(0.0f, 1.0f);
    Complex<float> D = A;

    if (use_single_task)
    {
        apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
    else
    {
        apply_gate(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
}

/**
 * @brief Applies the Z gate (phase flip) to a specific qubit in the state vector.
 * The Z gate introduces a phase of -1 to the |1⟩ state and leaves the |0⟩ state unchanged.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the Z gate is applied.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void z(sycl::queue &queue, Complex<float> *stateVector_d,
       const unsigned int numQubits,
       const int target, bool use_single_task)
{
    Complex<float> A(1.0f, 0.0f);
    Complex<float> B(0.0f, 0.0f);
    Complex<float> C = B;
    Complex<float> D(-1.0f, 0.0f);

    if (use_single_task)
    {
        apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
    else
    {
        apply_gate(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
}

/**
 * @brief Applies the RX rotation gate to a specific qubit in the state vector.
 * The RX gate rotates the qubit around the X axis by a specified angle.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the RX gate is applied.
 * @param angle The angle by which the qubit is rotated around the X axis.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void rx(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target,
        const double angle, bool use_single_task)
{
    double angle_2 = 0.5 * angle;
    double cos = std::cos(angle_2);
    double sin = std::sin(angle_2);

    Complex<float> A(cos, 0.0f);
    Complex<float> B(0.0f, -1.0 * sin);
    Complex<float> C(0.0f, -1.0 * sin);
    Complex<float> D(cos, 0.0f);

    if (use_single_task)
    {
        apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
    else
    {
        apply_gate(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
}

/**
 * @brief Applies the RY rotation gate to a specific qubit in the state vector.
 * The RY gate rotates the qubit around the Y axis by a specified angle.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the RY gate is applied.
 * @param angle The angle by which the qubit is rotated around the Y axis.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void ry(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target,
        const double angle, bool use_single_task)
{
    double angle_2 = 0.5 * angle;
    double cos = std::cos(angle_2);
    double sin = std::sin(angle_2);

    Complex<float> A(cos, 0.0f);
    Complex<float> B(-1.0 * sin, 0.0f);
    Complex<float> C(sin, 0.0f);
    Complex<float> D(cos, 0.0f);

    if (use_single_task)
    {
        apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
    else
    {
        apply_gate(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
}

/**
 * @brief Applies the RZ rotation gate to a specific qubit in the state vector.
 * The RZ gate applies a phase shift (exp(i * angle/2)) to the |1⟩ state while leaving the |0⟩ state unchanged.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target The target qubit on which the RZ gate is applied.
 * @param angle The angle by which the phase shift is applied to the |1⟩ state.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void rz(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target,
        const double angle, bool use_single_task)
{
    double angle_2 = 0.5 * angle;
    double cos = std::cos(angle_2);
    double sin = std::sin(angle_2);

    Complex<float> A(cos, -sin);
    Complex<float> B(0.0f, 0.0f);
    Complex<float> C(0.0f, 0.0f);
    Complex<float> D(cos, sin);

    if (use_single_task)
    {
        apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
    else
    {
        apply_gate(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
    }
}

/**
 * @brief Applies the CNOT gate to a 2 qubits in the state vector.
 * The CNOT applies a NOT on the 2nd qubit if the first is |1>.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target1 The first qubit on which the CNOT gate is applied.
 * @param target2 The second qubit on which the CNOT gate is applied.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void cnot(sycl::queue &queue, Complex<float> *stateVector_d,
          const unsigned int numQubits,
          const int target1, // control qubit
          const int target2, // target qubit
          bool use_single_task)
{
    Complex<float> AA(1.0f, 0.0f), AB(0.0f, 0.0f), AC(0.0f, 0.0f), AD(0.0f, 0.0f);
    Complex<float> BA(0.0f, 0.0f), BB(1.0f, 0.0f), BC(0.0f, 0.0f), BD(0.0f, 0.0f);
    Complex<float> CA(0.0f, 0.0f), CB(0.0f, 0.0f), CC(0.0f, 0.0f), CD(1.0f, 0.0f);
    Complex<float> DA(0.0f, 0.0f), DB(0.0f, 0.0f), DC(1.0f, 0.0f), DD(0.0f, 0.0f);

    if (use_single_task)
    {
        apply_gate2_single(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, AA, AB, AC, AD, BA, BB, BC, BD, CA, CB, CC, CD, DA, DB, DC, DD);
    }
    else
    {
        apply_gate2(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, AA, AB, AC, AD, BA, BB, BC, BD, CA, CB, CC, CD, DA, DB, DC, DD);
    }
}

/**
 * @brief Applies the SWAP gate to a 2 qubits in the state vector.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target1 The first qubit on which the SWAP gate is applied.
 * @param target2 The second qubit on which the SWAP gate is applied.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void swap(sycl::queue &queue, Complex<float> *stateVector_d,
          const unsigned int numQubits,
          const int target1, // control qubit
          const int target2, // target qubit
          bool use_single_task)
{
    Complex<float> AA(1.0f, 0.0f), AB(0.0f, 0.0f), AC(0.0f, 0.0f), AD(0.0f, 0.0f);
    Complex<float> BA(0.0f, 0.0f), BB(0.0f, 0.0f), BC(1.0f, 0.0f), BD(0.0f, 0.0f);
    Complex<float> CA(0.0f, 0.0f), CB(1.0f, 0.0f), CC(0.0f, 0.0f), CD(0.0f, 0.0f);
    Complex<float> DA(0.0f, 0.0f), DB(0.0f, 0.0f), DC(0.0f, 0.0f), DD(1.0f, 0.0f);

    if (use_single_task)
    {
        apply_gate2_single(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, AA, AB, AC, AD, BA, BB, BC, BD, CA, CB, CC, CD, DA, DB, DC, DD);
    }
    else
    {
        apply_gate2(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, AA, AB, AC, AD, BA, BB, BC, BD, CA, CB, CC, CD, DA, DB, DC, DD);
    }
}

/**
 * @brief Applies the Toffoli (CCNOT) gate to 3 qubits in the state vector.
 * The Toffoli gate applies a NOT on the target qubit if both control qubits are |1>.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target1 The first control qubit on which the Toffoli gate is applied.
 * @param target2 The second control qubit on which the Toffoli gate is applied.
 * @param target3 The target qubit on which the Toffoli gate is applied.
 * @param A, B, ..., CL The 64 parameters corresponding to the 8x8 matrix of the Toffoli gate.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void toffoli(sycl::queue &queue, Complex<float> *stateVector_d,
             const unsigned int numQubits,
             const int target1, const int target2, const int target3,
             bool use_single_task)
{
    // 8x8 matrix representation of the Toffoli gate is parameterized by the above 64 parameters.
    // Apply the Toffoli gate to the quantum state by using the apply_gate3 function

    Complex<float> AA(1.0f, 0.0f), AB(0.0f, 0.0f), AC(0.0f, 0.0f), AD(0.0f, 0.0f), AE(0.0f, 0.0f), AF(0.0f, 0.0f), AG(0.0f, 0.0f), AH(0.0f, 0.0f);
    Complex<float> BA(0.0f, 0.0f), BB(1.0f, 0.0f), BC(0.0f, 0.0f), BD(0.0f, 0.0f), BE(0.0f, 0.0f), BF(0.0f, 0.0f), BG(0.0f, 0.0f), BH(0.0f, 0.0f);
    Complex<float> CA(0.0f, 0.0f), CB(0.0f, 0.0f), CC(1.0f, 0.0f), CD(0.0f, 0.0f), CE(0.0f, 0.0f), CF(0.0f, 0.0f), CG(0.0f, 0.0f), CH(0.0f, 0.0f);
    Complex<float> DA(0.0f, 0.0f), DB(0.0f, 0.0f), DC(0.0f, 0.0f), DD(1.0f, 0.0f), DE(0.0f, 0.0f), DF(0.0f, 0.0f), DG(0.0f, 0.0f), DH(0.0f, 0.0f);
    Complex<float> EA(0.0f, 0.0f), EB(0.0f, 0.0f), EC(0.0f, 0.0f), ED(0.0f, 0.0f), EE(1.0f, 0.0f), EF(0.0f, 0.0f), EG(0.0f, 0.0f), EH(0.0f, 0.0f);
    Complex<float> FA(0.0f, 0.0f), FB(0.0f, 0.0f), FC(0.0f, 0.0f), FD(0.0f, 0.0f), FE(0.0f, 0.0f), FF(1.0f, 0.0f), FG(0.0f, 0.0f), FH(0.0f, 0.0f);
    Complex<float> GA(0.0f, 0.0f), GB(0.0f, 0.0f), GC(0.0f, 0.0f), GD(0.0f, 0.0f), GE(0.0f, 0.0f), GF(0.0f, 0.0f), GG(0.0f, 0.0f), GH(1.0f, 0.0f);
    Complex<float> HA(0.0f, 0.0f), HB(0.0f, 0.0f), HC(0.0f, 0.0f), HD(0.0f, 0.0f), HE(0.0f, 0.0f), HF(0.0f, 0.0f), HG(1.0f, 0.0f), HH(0.0f, 0.0f);

    if (use_single_task)
    {
        apply_gate3_single(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, target3,
                           AA, AB, AC, AD, AE, AF, AG, AH,
                           BA, BB, BC, BD, BE, BF, BG, BH,
                           CA, CB, CC, CD, CE, CF, CG, CH,
                           DA, DB, DC, DD, DE, DF, DG, DH,
                           EA, EB, EC, ED, EE, EF, EG, EH,
                           FA, FB, FC, FD, FE, FF, FG, FH,
                           GA, GB, GC, GD, GE, GF, GG, GH,
                           HA, HB, HC, HD, HE, HF, HG, HH);
    }
    else
    {
        apply_gate3(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, target3,
                    AA, AB, AC, AD, AE, AF, AG, AH,
                    BA, BB, BC, BD, BE, BF, BG, BH,
                    CA, CB, CC, CD, CE, CF, CG, CH,
                    DA, DB, DC, DD, DE, DF, DG, DH,
                    EA, EB, EC, ED, EE, EF, EG, EH,
                    FA, FB, FC, FD, FE, FF, FG, FH,
                    GA, GB, GC, GD, GE, GF, GG, GH,
                    HA, HB, HC, HD, HE, HF, HG, HH);
    }
}

/**
 * @brief Applies the controlledPhaseFlip (CCZ) gate to 3 qubits in the state vector.
 * The Toffoli gate applies a NOT on the target qubit if both control qubits are |1>.
 *
 * @param queue The SYCL queue for parallel execution.
 * @param stateVector_d The device memory containing the quantum state vector.
 * @param numQubits The number of qubits in the quantum state.
 * @param target1 The first control qubit on which the Toffoli gate is applied.
 * @param target2 The second control qubit on which the Toffoli gate is applied.
 * @param target3 The target qubit on which the Toffoli gate is applied.
 * @param A, B, ..., CL The 64 parameters corresponding to the 8x8 matrix of the Toffoli gate.
 * @param use_single_task Boolean used to know whether to apply a single_task or a parallel_for kernel.
 */
void controlledPhaseFlip(sycl::queue &queue, Complex<float> *stateVector_d,
                         const unsigned int numQubits,
                         const int target1, const int target2, const int target3,
                         bool use_single_task)
{
    // 8x8 matrix representation of the Toffoli gate is parameterized by the above 64 parameters.
    // Apply the Toffoli gate to the quantum state by using the apply_gate3 function

    Complex<float> AA(1.0f, 0.0f), AB(0.0f, 0.0f), AC(0.0f, 0.0f), AD(0.0f, 0.0f), AE(0.0f, 0.0f), AF(0.0f, 0.0f), AG(0.0f, 0.0f), AH(0.0f, 0.0f);
    Complex<float> BA(0.0f, 0.0f), BB(1.0f, 0.0f), BC(0.0f, 0.0f), BD(0.0f, 0.0f), BE(0.0f, 0.0f), BF(0.0f, 0.0f), BG(0.0f, 0.0f), BH(0.0f, 0.0f);
    Complex<float> CA(0.0f, 0.0f), CB(0.0f, 0.0f), CC(1.0f, 0.0f), CD(0.0f, 0.0f), CE(0.0f, 0.0f), CF(0.0f, 0.0f), CG(0.0f, 0.0f), CH(0.0f, 0.0f);
    Complex<float> DA(0.0f, 0.0f), DB(0.0f, 0.0f), DC(0.0f, 0.0f), DD(1.0f, 0.0f), DE(0.0f, 0.0f), DF(0.0f, 0.0f), DG(0.0f, 0.0f), DH(0.0f, 0.0f);
    Complex<float> EA(0.0f, 0.0f), EB(0.0f, 0.0f), EC(0.0f, 0.0f), ED(0.0f, 0.0f), EE(1.0f, 0.0f), EF(0.0f, 0.0f), EG(0.0f, 0.0f), EH(0.0f, 0.0f);
    Complex<float> FA(0.0f, 0.0f), FB(0.0f, 0.0f), FC(0.0f, 0.0f), FD(0.0f, 0.0f), FE(0.0f, 0.0f), FF(1.0f, 0.0f), FG(0.0f, 0.0f), FH(0.0f, 0.0f);
    Complex<float> GA(0.0f, 0.0f), GB(0.0f, 0.0f), GC(0.0f, 0.0f), GD(0.0f, 0.0f), GE(0.0f, 0.0f), GF(0.0f, 0.0f), GG(1.0f, 0.0f), GH(0.0f, 0.0f);
    Complex<float> HA(0.0f, 0.0f), HB(0.0f, 0.0f), HC(0.0f, 0.0f), HD(0.0f, 0.0f), HE(0.0f, 0.0f), HF(0.0f, 0.0f), HG(0.0f, 0.0f), HH(-1.0f, 0.0f);

    if (use_single_task)
    {
        apply_gate3_single(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, target3,
                           AA, AB, AC, AD, AE, AF, AG, AH,
                           BA, BB, BC, BD, BE, BF, BG, BH,
                           CA, CB, CC, CD, CE, CF, CG, CH,
                           DA, DB, DC, DD, DE, DF, DG, DH,
                           EA, EB, EC, ED, EE, EF, EG, EH,
                           FA, FB, FC, FD, FE, FF, FG, FH,
                           GA, GB, GC, GD, GE, GF, GG, GH,
                           HA, HB, HC, HD, HE, HF, HG, HH);
    }
    else
    {
        apply_gate3(queue, stateVector_d, std::pow(2, numQubits - 1), target1, target2, target3,
                    AA, AB, AC, AD, AE, AF, AG, AH,
                    BA, BB, BC, BD, BE, BF, BG, BH,
                    CA, CB, CC, CD, CE, CF, CG, CH,
                    DA, DB, DC, DD, DE, DF, DG, DH,
                    EA, EB, EC, ED, EE, EF, EG, EH,
                    FA, FB, FC, FD, FE, FF, FG, FH,
                    GA, GB, GC, GD, GE, GF, GG, GH,
                    HA, HB, HC, HD, HE, HF, HG, HH);
    }
}