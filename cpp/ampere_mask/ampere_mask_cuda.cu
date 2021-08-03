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
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mask
    ) {
        int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

        float a = input[x][y];
        int a_x = x;
        int a_y = y;

        float b = input[x+1][y];
        int b_x = x+1;
        int b_y = y;

        float c = input[x][y+1];
        int c_x = x;
        int c_y = y+1;

        float d = input[x+1][y+1];
        int d_x = x+1;
        int d_y = y+1;

        float tmp;
        int tmp_x = -1;
        int tmp_y = -1;

        if(a > b) { tmp = a; tmp_x = a_x; tmp_y = a_y;
                    a = b; a_x = b_x; a_y = b_y;
                    b = tmp; b_x = tmp_x; b_y = tmp_y;}
        if(c > d) { tmp = c; tmp_x = c_x; tmp_y = c_y;
                    c = d; c_x = d_x; c_y = d_y;
                    d = tmp; d_x = tmp_x; c_y = tmp_y;}
        if(a > c) { tmp = a; tmp_x = a_x; tmp_y = a_y;
                    a = c; a_x = c_x; a_y = c_y;
                    c = tmp; c_x = tmp_x; c_y = tmp_y;}
        if(b > d) { tmp = b; tmp_x = b_x; tmp_y = b_y;
                    b = d; b_x = d_x; b_y = d_y;
                    d = tmp; d_x = tmp_x; d_y = tmp_y;}
        if(b > c) { tmp = b; tmp_x = b_x; tmp_y = b_y;
                    b = c; b_x = c_x; b_y = c_y;
                    c = tmp; c_x = tmp_x; c_y = tmp_y;}
        // __syncthreads();
        // input[a_x][a_y] = 0;
        // input[b_x][b_y] = 0;
        // __syncthreads();
        mask[c_x][c_y] = 1;
        mask[d_x][d_y] = 1;
    }
}//namespace

// Assume input is flat
torch::Tensor ampere_mask_cuda(const torch::Tensor input) {
    // const int h = input.size(0);
    // const int w = input.size(1);

    // int pad_h = 0;
    // int pad_w = 0;

    // if(input.size(0) % 4 != 0) {pad_h = 4 - input.size(0) % 4;}
    // if(input.size(1) % 4 != 0) {pad_w = 4 - input.size(1) % 4;}

    // auto padded_input = F::pad(input.clone(), F::PadFuncOptions({0,pad_w,0,pad_h}));
    // auto padded_shape = input.size();
    // auto padded_input = padded_input.flatten()

    // 4x4 elements covered by one block
    const dim3 block_size(2,2);
    auto mask = torch::zeros_like(input);
    const dim3 blocks(input.size(0) / 4, input.size(1) / 4);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "ampere_cuda_kernel", ([&] {
        ampere_cuda_kernel<scalar_t><<<blocks, block_size>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
        );
    }));
    // mask = mask.index({Slice(None, h), Slice(None, w)});
    return mask;
}