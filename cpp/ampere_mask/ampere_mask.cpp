#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor ampere_mask_cuda(const torch::Tensor input);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// TODO: Add checks for number of dims and size of dims
// For now assume tensor comes in flat and padded to be divisible by 4
//#define CHECK_SIZE(x) TORCH_CHECK(x.size(0) % 4 == 0, #x " must have length divisible by 4")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ampere_mask(const torch::Tensor input) {
    CHECK_INPUT(input);
    return ampere_mask_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ampere_mask_cuda", &ampere_mask, "Apply Ampere sparsity to tensor (CUDA)");
}