#include "hmd.h"
#include "utils.h" 
#define LOG2(x) std::ceil(std::log(static_cast<double>(x)) / std::log(2.0))

void hmd_kernel_wrapper(int B, int outSize, int inSize, int L, const float *inputs, float *mem);

void hmd_grad_kernel_wrapper(int B, int outSize, int inSize, int L, const float *grads, 
                                const float *mem, float* lower);

at::Tensor hamilton_product(at::Tensor inputs){
    // inputs: (Batch, OutSize, InSize, 4) 
    CHECK_CONTIGUOUS(inputs);
    CHECK_IS_FLOAT(inputs);
    if (inputs.type().is_cuda()) {
        CHECK_CUDA(inputs);
    }
    else{
        TORCH_CHECK(false, "CPU not supported");
    }
    int inSize = inputs.size(2);
    int L = LOG2(inSize);
    at::Tensor mem = torch::zeros({inputs.size(0), inputs.size(1), 1<<(L+1), 4},
            at::device(inputs.device()).dtype(at::ScalarType::Float));

    hmd_kernel_wrapper(inputs.size(0), inputs.size(1), inSize, L, inputs.data<float>(), mem.data<float>());

    // mem: (Batch, OutSize, MemSize, 4) = MemSize = 2^( ceil(log2(InSize)) + 1)
    return mem;
    // result = mem[...,1,:], (Batch, OutSize, 4)
}

at::Tensor hamilton_product_grad(at::Tensor grads, at::Tensor mem, int inSize){
    // grads:    (Batch, OutSize, 4)
    // mem:      (Batch, OutSize, MemSize, 4)
    CHECK_CONTIGUOUS(grads);
    CHECK_IS_FLOAT(grads);
    CHECK_CONTIGUOUS(mem);
    CHECK_IS_FLOAT(mem);
    if (grads.type().is_cuda()&&mem.type().is_cuda()) {
        CHECK_CUDA(grads);
        CHECK_CUDA(mem);
    }
    else{
        TORCH_CHECK(false, "CPU not supported");
    }

    at::Tensor lower = torch::zeros({grads.size(0), grads.size(1), inSize, 4},
            at::device(grads.device()).dtype(at::ScalarType::Float));

    hmd_grad_kernel_wrapper(grads.size(0), grads.size(1), inSize, LOG2(inSize), grads.data<float>(), mem.data<float>(), lower.data<float>());
    return lower;
    // lower: (Batch, OutSize, InSize, 4) 
}