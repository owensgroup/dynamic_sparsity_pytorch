#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <nppdefs.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


using namespace torch::indexing;
namespace F = torch::nn::functional;

namespace {
    template <typename scalar_t>
    __global__ void ampere_cuda_kernel(
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input,
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mask
    ) {
        int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        float first = -NPP_MAXABS_32F;
        int idx_one = -1;
        int idx_two = -1;
        float second = -NPP_MAXABS_32F;
        for(int i = tid; i < tid+4; i++) {
            if(input[i] > first) {
                second = first;
                idx_two = idx_one;

                first = input[i];
                idx_one = i;
            }
            else if(input[i] > second) {
                second = input[i];
                idx_two = i;
            }
        }
        mask[idx_one] = 1;
        mask[idx_two] = 1;
    }
}//namespace

// Assume input is flat
torch::Tensor ampere_mask_cuda(const torch::Tensor input) {
    // const int pad_h = input.size(2) % 4;
    // const int pad_w = input.size(3) % 4;
    // auto padded_input = F::pad(input.clone(), F::PadFuncOptions({0,pad_w,0,pad_h}));
    // auto padded_shape = input.size();
    // auto padded_input = padded_input.flatten()
    const int num_el = input.size(0);
    const int pad_len = 4 - num_el % 4;
    auto padded_input = F::pad(input.clone(), F::PadFuncOptions({0,pad_len}));
    auto mask = torch::zeros_like(padded_input);
    const int block_size = 256;
    const dim3 blocks((num_el / 4 + block_size - 1) / block_size);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "ampere_cuda_kernel", ([&] {
        ampere_cuda_kernel<scalar_t><<<blocks, block_size>>>(
            padded_input.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
        );
    }));
    mask = mask.index({Slice(None, num_el)});
    return mask;
}