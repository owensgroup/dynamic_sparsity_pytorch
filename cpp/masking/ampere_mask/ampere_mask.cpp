#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor ampere_mask_cuda(torch::Tensor input);
torch::Tensor batched_amp_mask_seq(torch::Tensor input);
torch::Tensor batched_amp_mask_stream(torch::Tensor input);
torch::Tensor batched_amp_mask_strided(torch::Tensor input);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) TORCH_CHECK(x.dim() <= 4, #x " tensor must have rank less than 4")


#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIMS(x)

torch::Tensor ampere_mask(torch::Tensor input) {
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

torch::Tensor batched_ampere_mask(torch::Tensor input) {
    CHECK_INPUT(input);
    return batched_amp_mask_strided(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ampere", &ampere_mask, "Apply Ampere sparsity to tensor (CUDA)");
    m.def("batched_ampere", &batched_ampere_mask,
            "Apply Ampere sparsity to batch of images (CUDA)");
}