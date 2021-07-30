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
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input
    ) {
        int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        
        
        float first = NPP_MAXABS_32F;
        int idx_one_x = -1;
        int idx_one_y = -1;
        int idx_two_x = -1;
        int idx_two_y = -1;
        float second = NPP_MAXABS_32F;


        for(int i = x; i < x+2; i++) {
            for(int j = y; j < y+2; j++) {
                if(input[i][j] < first) {
                    second = first;
                    idx_two_x = idx_one_x;
                    idx_two_y = idx_one_y;

                    first = input[i][j];
                    idx_one_x = i;
                    idx_one_y = j;
                }
                else if(input[i][j] < second) {
                    second = input[i][j];
                    idx_two_x = i;
                    idx_two_y = j;
                }
            }
        }

        input[idx_two_x][idx_two_y] = 0;
        input[idx_one_x][idx_one_y] = 0;
    }
}//namespace

// Assume input is flat and padded already
torch::Tensor ampere_mask_cuda(const torch::Tensor input) {
    // const int pad_h = 4 - input.size(2) % 4;
    // const int pad_w = 4 - input.size(3) % 4;
    // auto padded_input = F::pad(input.clone(), F::PadFuncOptions({0,pad_w,0,pad_h}));
    // auto padded_shape = input.size();
    // auto padded_input = padded_input.flatten()
    // const int num_el = input.size(0);
    // const int pad_len = 4 - num_el % 4;
    // auto padded_input = F::pad(input.clone(), F::PadFuncOptions({0,pad_len}));
    auto mask = torch::zeros_like(input);

    // Each block reads in 16 rows and 16 columns
    const dim3 block_size(8, 8);
    int gridDim_x = input.size(0) / 8;
    int gridDim_y = input.size(1) / 8;
    const dim3 blocks(gridDim_x, gridDim_y);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "ampere_cuda_kernel", ([&] {
        ampere_cuda_kernel<scalar_t><<<blocks, block_size>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
        );
    }));

    return input;
}