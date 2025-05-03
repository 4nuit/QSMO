/**
 * @file kernels.cpp
 */

 #include "../headers/kernels.hpp"
 #include <bitset>
 #include <random>
 #include <algorithm>
 #include <numeric>
 #include <string>
 
 // #include <oneapi/math.hpp>
 // #include <oneapi/mkl/vm.hpp>
 
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
         #pragma unroll 2
         for (global_id = 0; global_id < numStates; ++global_id)
         {
             const int zero_state = nth_cleared(global_id, target);
             const int one_state = zero_state | (1 << target);
             Complex<float> zero_amp = stateVector_d[zero_state];
             Complex<float> one_amp = stateVector_d[one_state];
             stateVector_d[zero_state] = A * zero_amp + B * one_amp;
             stateVector_d[one_state] = C * zero_amp + D * one_amp;
         }
     });
     event.wait();
     auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
     auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
     std::cout << "Apply gate1_single elapsed time: " << (end - start) / 1.0e9 << "seconds\n";
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
 void get_proba_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates, float *probs_d)
 {
     queue.single_task<class Proba_single>([=]()
     {
         int global_id;
         #pragma unroll 1
         for(global_id = 0; global_id < numStates; ++global_id){
             Complex<float> amp = stateVector_d[global_id];
             probs_d[global_id] = Absol_v(amp * amp);
         }
     }).wait();
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
 
     get_proba_single(queue, stateVector_d, numStates, probs_d);
  
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
 
     apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
 }
 
 void h_n(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates) {}
 
 void blas_h(sycl::queue &queue, std::complex<float> *stateVector_d, const unsigned int numStates, int target) {}
 
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
 
     apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
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
 
     apply_gate_single(queue, stateVector_d, std::pow(2, numQubits - 1), target, A, B, C, D);
 }
 
 void y(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target, bool use_single_task) {}
 
 void rx(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target, const double angle, bool use_single_task) {}
 
 void ry(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target, const double angle, bool use_single_task) {}
 
 void rz(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target, const double angle, bool use_single_task) {}
 
 void cnot(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target1, const int target2, bool use_single_task) {}
 
 void swap(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target1, const int target2, bool use_single_task) {}
 
 void toffoli(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numQubits, const int target1, const int target2, const int target3, bool use_single_task) {}
 
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
     auto event =  queue.single_task<class Gate3_single>([=]()
     {
         int numStates = 1 << (numQubits-1);
         int global_id;
         #pragma unroll 2
         for(global_id = 0; global_id < numStates; ++global_id)
         {
             int state_000 = nth_cleared(global_id, target1) & nth_cleared(global_id, target2) & nth_cleared(global_id, target3);
             int state_001 = state_000 | (1 << target1);  // Set the 1st qubit to 1
             //int state_010 = state_000 | (1 << target2);  // Set the 2nd qubit to 1
             int state_011 = state_001 | (1 << target2);  // Set both the 1st and 2nd qubits to 1
             //int state_100 = state_000 | (1 << target3);  // Set the 3rd qubit to 1
             //int state_101 = state_001 | (1 << target3);  // Set the 1st and 3rd qubits to 1
             //int state_110 = state_010 | (1 << target3);  // Set the 2nd and 3rd qubits to 1
             int state_111 = state_011 | (1 << target3);  // Set all 3 qubits to 1
 
             stateVector_d[state_111] *= -1;
         }
     });
     event.wait();
     auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
     auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
     std::cout << "Apply gate3_single elapsed time: " << (end - start) / 1.0e9 << " seconds\n";
}
