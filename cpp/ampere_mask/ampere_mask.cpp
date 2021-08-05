#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor ampere_mask_cuda(torch::Tensor input);
torch::Tensor batched_amp_mask_seq(torch::Tensor input);
torch::Tensor batched_amp_mask_stream(torch::Tensor input);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) TORCH_CHECK(x.dim() <= 4, #x " tensor must have rank less than 4")

// TODO: Add checks for number of dims and size of dims
// For now assume tensor comes in flat and padded to be divisible by 4
//#define CHECK_SIZE(x) TORCH_CHECK(x.size(0) % 4 == 0, #x " must have length divisible by 4")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIMS(x)

torch::Tensor ampere_mask(torch::Tensor input, bool use_streams=false) {
    CHECK_INPUT(input);
    if(input.dim() == 3) {
        if(use_streams) {
            return batched_amp_mask_stream(input);
        }
        else {
            return batched_amp_mask_seq(input);
        }
    }
    if(input.dim() == 2) {
        return ampere_mask_cuda(input);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ampere", &ampere_mask, "Apply Ampere sparsity to tensor (CUDA)");
}