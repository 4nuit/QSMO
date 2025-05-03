/*******************************************************************************
 * Copyright 2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

/*
 *
 *  Content:
 *       This example demonstrates use of DPCPP API oneapi::math::blas::gemm
 *       using unified shared memory to perform General Matrix-Matrix
 *       Multiplication on a SYCL device (HOST, CPU, GPU) that is selected
 *       during runtime.
 *
 *       C = alpha * op(A) * op(B) + beta * C
 *
 *       where op() is defined by one of oneapi::math::transpose::{nontrans,trans,conjtrans}
 *
 *
 *       This example demonstrates only single precision (float) data type for
 *       gemm matrix data
 *
 *
 *******************************************************************************/

// stl includes
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <oneapi/math.hpp>
#include "../headers/example_helper.hpp"

#if CPU
    #include <oneapi/mkl/vm.hpp>
    #define ONEAPI oneapi::math::blas::mklcpu::column_major //oneapi::mkl::sparse 
    #define ONEAPI_ROW oneapi::math::blas::mklcpu::row_major
#elif GPU
    #include <oneapi/math/blas/detail/cublas/onemath_blas_cublas.hpp>
    #define ONEAPI oneapi::math::blas::cublas::column_major
    #define ONEAPI_ROW oneapi::math::blas::cublas::row_major
#else
    #include <oneapi/math/blas/detail/generic/onemath_blas_generic.hpp>
    #define ONEAPI oneapi::math::blas::generic::column_major
    #define ONEAPI_ROW oneapi::math::blas::generic::row_major
#endif

//
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
void run_gemm_example(const sycl::device &dev)
{
    //
    // Initialize data for Gemm
    //
    // C = alpha * op(A) * op(B)  + beta * C
    //

    oneapi::math::transpose transA = oneapi::math::transpose::nontrans;
    oneapi::math::transpose transB = oneapi::math::transpose::nontrans;

    // matrix data sizes
    int m = 2;
    int n = 2;
    int k = 2;

    // leading dimensions of data
    int ldA = 2;
    int ldB = 2;
    int ldC = 2;
    int sizea = (transA == oneapi::math::transpose::nontrans) ? ldA * k : ldA * m;
    int sizeb = (transB == oneapi::math::transpose::nontrans) ? ldB * n : ldB * k;
    int sizec = ldC * n;

    // set scalar fp values
    float alpha = 1;//set_fp_value(float(1.0), float(0.0));
    float beta = 0;//set_fp_value(float(0.0), float(0.0));

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions)
    {
        for (std::exception_ptr const &e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e)
            {
                std::cerr << "Caught asynchronous SYCL exception during GEMM:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    // create execution queue
    sycl::queue main_queue(dev, exception_handler);
    sycl::event gemm_done;
    sycl::context cxt = main_queue.get_context();

    auto device = main_queue.get_device();
    std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

    // allocate matrix on host
    std::vector<float> A(sizea);
    std::vector<float> B(sizeb);
    std::vector<float> C(sizec);
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(C.begin(), C.end(), 0); //If beta = 0, matrix C does not need to be initialized before calling gemm.

    //rand_matrix(A, transA, m, k, ldA);
    //rand_matrix(B, transB, k, n, ldB);
    //rand_matrix(C, oneapi::math::transpose::nontrans, m, n, ldC);
    A.at(0) = 1;
    A.at(1) = 2;
    A.at(2) = 3;
    A.at(3) = 4;
    B.at(0) = 5;
    B.at(1) = 6;
    B.at(2) = 7;
    B.at(3) = 8;

    // allocate memory on device
    auto dev_A = sycl::malloc_device<float>(sizea * sizeof(float), main_queue);
    auto dev_B = sycl::malloc_device<float>(sizeb * sizeof(float), main_queue);
    auto dev_C = sycl::malloc_device<float>(sizec * sizeof(float), main_queue);
    if (!dev_A || !dev_B || !dev_C)
    {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    // copy data from host to device
    main_queue.memcpy(dev_A, A.data(), sizea * sizeof(float)).wait();
    main_queue.memcpy(dev_B, B.data(), sizeb * sizeof(float)).wait();
    main_queue.memcpy(dev_C, C.data(), sizec * sizeof(float)).wait();

    //
    // Execute Gemm
    //
    // add oneapi::math::blas::gemm to execution queue
    // oneapi::math::blas::generic::column_major
    gemm_done = ONEAPI::gemm(main_queue, transA, transB, m, n, k, alpha, dev_A, ldA, dev_B, ldB, beta, dev_C, ldC);

    // Wait until calculations are done
    gemm_done.wait_and_throw();
    auto end = gemm_done.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = gemm_done.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cout << "Apply gate elapsed time: " << (end - start) / 1.0e6 << "ms\n";

    //
    // Post Processing
    //
    // copy data from device back to host
    main_queue.memcpy(C.data(), dev_C, sizec * sizeof(float)).wait_and_throw();

    std::cout << "\n\t\tGEMM parameters:" << std::endl;
    std::cout << "\t\t\ttransA = "
              << (transA == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (transA == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << ", transB = "
              << (transB == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (transB == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << ", k = " << k << std::endl;
    std::cout << "\t\t\tlda = " << ldA << ", ldB = " << ldB << ", ldC = " << ldC << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;

    std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(A.data(), ldA, "A");

    // output the top 2x2 block of B matrix
    print_2x2_matrix_values(B.data(), ldB, "B");

    // output the top 2x2 block of C matrix
    print_2x2_matrix_values(C.data(), ldC, "C");

    sycl::free(dev_C, main_queue);
    sycl::free(dev_B, main_queue);
    sycl::free(dev_A, main_queue);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner()
{
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# General Matrix-Matrix Multiplication using Unified Shared Memory Example: "
              << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# C = alpha * A * B + beta * C" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A, B and C are general dense matrices and alpha, beta are" << std::endl;
    std::cout << "# floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   gemm" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Device will be selected during runtime." << std::endl;
    std::cout << "# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify"
              << std::endl;
    std::cout << "# available devices" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example
//
int main(int argc, char **argv)
{
    try
    {
        print_example_banner();

#if FPGA_SIMULATOR
        auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
        auto selector = sycl::ext::intel::fpga_selector_v;
#elif FPGA_EMULATOR
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif GPU
        auto selector = sycl::gpu_selector_v;
#else
        auto selector = sycl::cpu_selector_v;
#endif

        run_gemm_example((const sycl::device)selector);
        std::cout << "BLAS GEMM USM example ran OK." << std::endl;
    }
    catch (sycl::exception const &e)
    {
        std::cerr << "Caught synchronous SYCL exception during GEMM:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << "Caught std::exception during GEMM:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}