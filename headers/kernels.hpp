#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#if GPU
        #include <thrust/complex.h>
        template <typename T>
        using Complex = thrust::complex<T>;
        #define Absol_v thrust::abs
        #define Arg_v thrust::arg
#else
        #include <complex>
        #include <cmath>
        template <typename T>
        using Complex = std::complex<T>;
        #define Absol_v std::abs
        #define Arg_v std::arg
#endif

/*
#if CPU
        #include <oneapi/mkl/vm.hpp>
        #define COLUMN_MAJOR oneapi::math::blas::column_major
        #define ROW_MAJOR oneapi::math::blas::row_major
#elif GPU
        #include <oneapi/math/blas/detail/cublas/onemath_blas_cublas.hpp>
        #define COLUMN_MAJOR oneapi::math::blas::cublas::column_major
        #define ROW_MAJOR oneapi::math::blas::cublas::row_major
#else
        #include <oneapi/math/blas/detail/generic/onemath_blas_generic.hpp>
        #define COLUMN_MAJOR oneapi::math::blas::generic::column_major
        #define ROW_MAJOR oneapi::math::blas::cublas::row_major
#endif
*/

std::string toBinary(int num);

SYCL_EXTERNAL int nth_cleared(int n, int target);

/************************* BLASCIRCUIT CLASS ************************/

void blas_h(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target);

void apply_spmv_gate(sycl::queue &queue, float *stateVector_d, int numStates, float *y, int *ia, int *ja, float *A, int size, int nnz, float alpha, float beta, int target);

void sparse_x(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target);

void sparse_x_all(sycl::queue &queue, float *stateVector_d, float *y, const unsigned int numQubits, float alpha, float beta);

void sparse_z(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target);

void sparse_ccnot_all(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta);

void sparse_ccflip(sycl::queue &queue, float *stateVector_d, const unsigned int numQubits, float alpha, float beta, int target1, int target2, int target3);

void blas_measure(sycl::queue &queue, float *stateVector_d, int numQubits, int samples);

/************************* CIRCUIT CLASS ************************/

void apply_gate(sycl::queue &queue, Complex<float> *stateVector_d,
                const unsigned int numStates,
                const int target,
                const Complex<float> A,
                const Complex<float> B,
                const Complex<float> C,
                const Complex<float> D);

void apply_gate_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                const int target,
                const Complex<float> A,
                const Complex<float> B,
                const Complex<float> C,
                const Complex<float> D);

void apply_gate2(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                const int target1, const int target2,
                const Complex<float> A, const Complex<float> B, const Complex<float> C, const Complex<float> D,
                const Complex<float> E, const Complex<float> F, const Complex<float> G, const Complex<float> H,
                const Complex<float> I, const Complex<float> J, const Complex<float> K, const Complex<float> L,
                const Complex<float> M, const Complex<float> N, const Complex<float> O, const Complex<float> P);

void apply_gate2_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                const int target1, const int target2,
                const Complex<float> A, const Complex<float> B, const Complex<float> C, const Complex<float> D,
                const Complex<float> E, const Complex<float> F, const Complex<float> G, const Complex<float> H,
                const Complex<float> I, const Complex<float> J, const Complex<float> K, const Complex<float> L,
                const Complex<float> M, const Complex<float> N, const Complex<float> O, const Complex<float> P);

void apply_gate3(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                const int target1, const int target2, const int target3,
                const Complex<float> AA, const Complex<float> AB, const Complex<float> AC, const Complex<float> AD, const Complex<float> AE, const Complex<float> AF, const Complex<float> AG, const Complex<float> AH,
                const Complex<float> BA, const Complex<float> BB, const Complex<float> BC, const Complex<float> BD, const Complex<float> BE, const Complex<float> BF, const Complex<float> BG, const Complex<float> BH,
                const Complex<float> CA, const Complex<float> CB, const Complex<float> CC, const Complex<float> CD, const Complex<float> CE, const Complex<float> CF, const Complex<float> CG, const Complex<float> CH,
                const Complex<float> DA, const Complex<float> DB, const Complex<float> DC, const Complex<float> DD, const Complex<float> DE, const Complex<float> DF, const Complex<float> DG, const Complex<float> DH,
                const Complex<float> EA, const Complex<float> EB, const Complex<float> EC, const Complex<float> ED, const Complex<float> EE, const Complex<float> EF, const Complex<float> EG, const Complex<float> EH,
                const Complex<float> FA, const Complex<float> FB, const Complex<float> FC, const Complex<float> FD, const Complex<float> FE, const Complex<float> FF, const Complex<float> FG, const Complex<float> FH,
                const Complex<float> GA, const Complex<float> GB, const Complex<float> GC, const Complex<float> GD, const Complex<float> GE, const Complex<float> GF, const Complex<float> GG, const Complex<float> GH,
                const Complex<float> HA, const Complex<float> HB, const Complex<float> HC, const Complex<float> HD, const Complex<float> HE, const Complex<float> HF, const Complex<float> HG, const Complex<float> HH);

void apply_gate3_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates,
                const int target1, const int target2, const int target3,
                const Complex<float> AA, const Complex<float> AB, const Complex<float> AC, const Complex<float> AD, const Complex<float> AE, const Complex<float> AF, const Complex<float> AG, const Complex<float> AH,
                const Complex<float> BA, const Complex<float> BB, const Complex<float> BC, const Complex<float> BD, const Complex<float> BE, const Complex<float> BF, const Complex<float> BG, const Complex<float> BH,
                const Complex<float> CA, const Complex<float> CB, const Complex<float> CC, const Complex<float> CD, const Complex<float> CE, const Complex<float> CF, const Complex<float> CG, const Complex<float> CH,
                const Complex<float> DA, const Complex<float> DB, const Complex<float> DC, const Complex<float> DD, const Complex<float> DE, const Complex<float> DF, const Complex<float> DG, const Complex<float> DH,
                const Complex<float> EA, const Complex<float> EB, const Complex<float> EC, const Complex<float> ED, const Complex<float> EE, const Complex<float> EF, const Complex<float> EG, const Complex<float> EH,
                const Complex<float> FA, const Complex<float> FB, const Complex<float> FC, const Complex<float> FD, const Complex<float> FE, const Complex<float> FF, const Complex<float> FG, const Complex<float> FH,
                const Complex<float> GA, const Complex<float> GB, const Complex<float> GC, const Complex<float> GD, const Complex<float> GE, const Complex<float> GF, const Complex<float> GG, const Complex<float> GH,
                const Complex<float> HA, const Complex<float> HB, const Complex<float> HC, const Complex<float> HD, const Complex<float> HE, const Complex<float> HF, const Complex<float> HG, const Complex<float> HH);
    

void get_proba(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates, float *probaVector_d);

void get_proba_single(sycl::queue &queue, Complex<float> *stateVector_d, const unsigned int numStates, float *probaVector_d);

void measure(sycl::queue &queue, Complex<float> *stateVector_d, int numQubits, int samples, bool use_single_task);

void h(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target, bool use_single_task);

void h_n(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numStates);

void h_n_single(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numStates);

void x(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target, bool use_single_task);

void y(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target, bool use_single_task);

void z(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target, bool use_single_task);

void rx(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target,
        const double angle, bool use_single_task);

void ry(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target,
        const double angle, bool use_single_task);

void rz(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target,
        const double angle, bool use_single_task);

void cnot(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target1, const int target2, bool use_single_task);

void swap(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target1, const int target2, bool use_single_task);

void toffoli(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target1, const int target2, const int target3, bool use_single_task);

void controlledPhaseFlip(sycl::queue &queue, Complex<float> *stateVector_d,
        const unsigned int numQubits,
        const int target1, const int target2, const int target3, bool use_single_task);

#endif