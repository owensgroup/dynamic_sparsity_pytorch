#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor ampere_prune_cu(torch::Tensor input);
torch::Tensor ampere_prune_batched(torch::Tensor input);

torch::Tensor random_prune(torch::Tensor input);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) TORCH_CHECK(x.dim() <= 4, #x " tensor must have rank less than 4")


#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DIMS(x)

torch::Tensor ampere_prune(torch::Tensor input) {
    CHECK_INPUT(input);

    // Single sample fed (CHW)
    if(input.dim() == 3) {
        return ampere_prune_cu(input);
    }
    //     else if(input.dim() == 4) {
        return ampere_prune_batched(input);
    }
}

torch::Tensor random_prune(torch::Tensor input) {
    CHECK_INPUT(input);
    auto inds = torch::randperm(input.numel()).index({Slice(None, input.numel()/2)});
    auto f = input.flatten().index(inds)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ampere", &ampere_prune, "Generate Ampere Sparse Mask for input tensor");
    m.def("unstructured", &random_prune, "Generate unstructured sparse mask for input tensor");
}