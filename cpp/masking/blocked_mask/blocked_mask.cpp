#include <torch/extension.h>
#include <iostream>

torch::Tensor blocked_mask_cuda(torch::Tensor input);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) TORCH_CHECK(x.dim() <= 3, #x " tensor must have rank less than 4")