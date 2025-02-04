#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <nppdefs.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <ATen/cuda/CUDAContext.h>

using namespace torch::indexing;
namespace F = torch::nn::functional;

namespace {
    template <typename scalar_t>
    __global__ void ampere_mask_kernel(
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input,
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mask
    ) {
        int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

        float a = input[tid];
        int idx_a = tid;

        float b = input[tid+1];
        int idx_b = tid+1;

        float c = input[tid+2];
        int idx_c = tid+2;

        float d = input[tid+3];
        int idx_d = tid+3;

        float tmp;
        int tmp_idx = -1;

        if(a > b) { tmp = a; tmp_idx = idx_a;
                  a = b; idx_a = idx_a;
                  b = tmp; idx_b = tmp_idx;}
        if(c > d) { tmp = c; tmp_idx = idx_c;
                  c = d; idx_c = idx_d;
                  d = tmp; idx_d = tmp_idx;}
        if(a > c) { tmp = a; tmp_idx = idx_a;
                    a = c; idx_a = idx_c;
                    c = tmp; idx_c = tmp_idx;}
        if(b > d) { tmp = b; tmp_idx = idx_b;
                    b = d; idx_b = idx_d;
                    d = tmp; d_idx = tmp_idx;}
        if(b > c) { tmp = b; tmp_idx = idx_b;
                    b = c; idx_b = idx_c;
                    c = tmp; idx_c = tmp_idx;}
        
        __syncthreads();
        mask[idx_c] = 1;
        mask[idx_d] = 1;
    }


    // template <typename scalar_t>
    // __global__ void batched_ampere_kernel(
    //     const torch::PackedTensorAccessor<scalar_t, 2,torch::RestrictPtrTraits,size_t> input,
    //     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> mask,
    //     const size_t flat_len,
    //     const size_t batch_size
    // ) {

    //     size_t num_iter = (batch_size - 1) / gridDim.y + 1;
    //     size_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    //     for(size_t y = blockIdx.y; n < num_iter * gridDim.y; n += gridDim.y) {
    //         if(x < flat_len && y < batch_size) {
    //             float a = input[y][x];
    //             int idx_a = x;

    //             float b = input[y][x+1];
    //             int idx_b = x+1;

    //             float c = input[y][x+2];
    //             int idx_c = x+2;

    //             float d = input[y][x+3];
    //             int idx_d = x+3;

    //             float tmp;
    //             int tmp_idx = -1;

    //             if(a > b) { tmp = a; tmp_idx = idx_a;
    //                 a = b; idx_a = idx_a;
    //                 b = tmp; idx_b = tmp_idx;}
    //             if(c > d) { tmp = c; tmp_idx = idx_c;
    //                     c = d; idx_c = idx_d;
    //                     d = tmp; idx_d = tmp_idx;}
    //             if(a > c) { tmp = a; tmp_idx = idx_a;
    //                         a = c; idx_a = idx_c;
    //                         c = tmp; idx_c = tmp_idx;}
    //             if(b > d) { tmp = b; tmp_idx = idx_b;
    //                         b = d; idx_b = idx_d;
    //                         d = tmp; d_idx = tmp_idx;}
    //             if(b > c) { tmp = b; tmp_idx = idx_b;
    //                         b = c; idx_b = idx_c;
    //                         c = tmp; idx_c = tmp_idx;}

    //             __syncthreads();
    //             mask[y][idx_c] = 1;
    //             mask[y][idx_d] = 1;
    //         }
    //     }
        
    // }

    template <typename scalar_t>
    __global__ void block2_cuda_kernel(
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
        /* Uncomment to also change input tensor */
        // input[a_x][a_y] = 0;
        // input[b_x][b_y] = 0;

        // Barrier before writes to mask, should increase throughput of write accesses,
        // through coalesced access vs random access from divergence
        __syncthreads();
        mask[c_x][c_y] = 1;
        mask[d_x][d_y] = 1;
    }


    template<typename scalar_t>
    __global__ void batched_block2_mask_kernel(
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> mask,
        const unsigned int batch_size,
        const unsigned int h,
        const unsigned int w
    ) {
        unsigned int n = blockIdx.z;
        unsigned int num_iter = (batch_size - 1) / gridDim.z + 1;

        unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        for(unsigned int n = blockIdx.z; n < num_iter * gridDim.z; n += gridDim.z) {
            if(n < batch_size && y < w && x < h) {
                float a = input[n][x][y];
                int a_x = x;
                int a_y = y;

                float b = input[n][x+1][y];
                int b_x = x+1;
                int b_y = y;

                float c = input[n][x][y+1];
                int c_x = x;
                int c_y = y+1;

                float d = input[n][x+1][y+1];
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
                
                __syncthreads();
                mask[n][c_x][c_y] = 1;
                mask[n][d_x][d_y] = 1;
            }
        }
        
    }
}//namespace

// Expects a rank 3 tensor (CHW)
// C - num channels
// H - height of img
// W - width of img
torch::Tensor ampere_mask_cuda(torch::Tensor input) {
    const auto n = input.size(0);
    const auto c = input.size(1);
    const auto h = input.size(2);
    const auto w = input.size(3);
    input = input.flatten();
    int64_t pad = 0;
    if(input.size(0) % 4 != 0) {pad = 4 - input.size(0) % 4;}

    auto padded_input = F::pad(input, F::PadFuncOptions({0, pad}));
    // Each thread covers 4 elements
    // This block loads 256 elements at once
    const dim3 block_size(64);
    const dim3 num_blocks((input.size(0) - 1) / 256 + 1);

    const size_t num_sms = 80;
    const size_t blocks_per_sm = 2048/64;

    const size_t total_blocks = 

}
torch::Tensor block2_mask_cuda(torch::Tensor input) {
    auto h = input.size(0);
    auto w = input.size(1);
    int64_t pad_h = 0;
    int64_t pad_w = 0;

    if(input.size(0) % 4 != 0) {pad_h = 4 - input.size(0) % 4;}
    if(input.size(1) % 4 != 0) {pad_w = 4 - input.size(1) % 4;}

    auto padded_input = F::pad(input, F::PadFuncOptions({0,pad_w,0,pad_h}));

    // 16x16 elements covered by one block
    // Each thread in a block processes 2 elements
    const dim3 block_size(8,8);
    const dim3 blocks((padded_input.size(0)-1) / 16 + 1, (padded_input.size(1)-1) / 16 + 1);

    auto mask = torch::zeros_like(padded_input);
    AT_DISPATCH_FLOATING_TYPES(padded_input.type(), "ampere_cuda_kernel", ([&] {
        ampere_cuda_kernel<scalar_t><<<blocks, block_size>>>(
            padded_input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
            );
        })
    );
    mask = mask.index({Slice(None, h), Slice(None, w)});
    return mask;
}

torch::Tensor batched_block2_mask_strided(torch::Tensor input) {
    const unsigned int batch_size = input.size(0);
    const auto h = input.size(1);
    const auto w = input.size(2);

    int64_t pad_h = 0;
    int64_t pad_w = 0;
    if(input.size(1) % 4 != 0) {pad_h = 4 - input.size(1) % 4;}
    if(input.size(2) % 4 != 0) {pad_w = 4 - input.size(2) % 4;}
    auto padded_input = F::pad(input, F::PadFuncOptions({0,pad_w,0,pad_h,0,0}));

    const uint64_t num_sms = 36;
    const dim3 block_size(8,8);


    // Threads per sm / threads per block
    const uint64_t blocks_per_sm = 1024 / 64;

    // Totals blocks running at once
    const uint64_t total_blocks = num_sms * blocks_per_sm;
    const auto blocks_x = (padded_input.size(1) - 1) / 16 + 1;
    const auto blocks_y = (padded_input.size(2) - 1) / 16 + 1;
    const auto blocks_per_img = blocks_x * blocks_y;
    
    
    const uint64_t img_per_sm = (float)blocks_per_sm / (float)blocks_per_img * num_sms;
    size_t blocks_z;
    if(blocks_per_img > total_blocks) {
        blocks_z = 1;
    }
    else if(batch_size < img_per_sm) {
        blocks_z = batch_size;
    }
    else {
        blocks_z = img_per_sm;
    }
    // printf("blocks_z: %d\n", blocks_z);
    const dim3 grid_size(blocks_x, blocks_y, blocks_z);
    
    auto mask = torch::zeros_like(padded_input);
    AT_DISPATCH_FLOATING_TYPES(padded_input.type(), "ampere_cuda_kernel", ([&] {
    batched_ampere_mask_kernel<scalar_t><<<grid_size, block_size>>>(
            padded_input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            batch_size,
            padded_input.size(1),
            padded_input.size(2)
        );
    }));

    mask = mask.index({Slice(None,None), Slice(None, h), Slice(None, w)});
    return mask;
}


torch::Tensor batched_block2_mask_seq(torch::Tensor input) {
    const int batch_size = input.size(0);
    auto h = input.size(1);
    auto w = input.size(2);
    int64_t pad_h = 0;
    int64_t pad_w = 0;

    if(input.size(1) % 4 != 0) {pad_h = 4 - input.size(1) % 4;}
    if(input.size(2) % 4 != 0) {pad_w = 4 - input.size(2) % 4;}

    

    auto padded_input = F::pad(input, F::PadFuncOptions({0,pad_w,0,pad_h,0,0}));

    const dim3 block_size(2, 2);
    const dim3 grid_size(padded_input.size(1)/4, padded_input.size(2)/4);

    torch::Tensor masks = torch::zeros_like(padded_input);
    for(int batch = 0; batch < batch_size; batch++) {
        auto img = padded_input[batch];
        auto mask = masks[batch];
        AT_DISPATCH_FLOATING_TYPES(img.type(), "ampere_cuda_kernel", ([&] {
        ampere_cuda_kernel<scalar_t><<<grid_size, block_size>>>(
            img.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
            );
        }));
    }
    masks = masks.index({Slice(None,None), Slice(None, h), Slice(None, w)});
    return masks;
}

torch::Tensor batched_block2_mask_stream(torch::Tensor input) {
    auto h = input.size(1);
    auto w = input.size(2);
    int64_t pad_h = 0;
    int64_t pad_w = 0;

    if(input.size(1) % 4 != 0) {pad_h = 4 - input.size(1) % 4;}
    if(input.size(2) % 4 != 0) {pad_w = 4 - input.size(2) % 4;}
    const int batch_size = input.size(0);
    const int sample_per_stream = batch_size / 32;
    
    auto padded_input = F::pad(input, F::PadFuncOptions({0,pad_w,0,pad_h,0,0}));
    const dim3 block_size(2, 2);
    const dim3 grid_size(padded_input.size(1)/4, padded_input.size(2)/4);

    torch::Tensor masks = torch::zeros_like(padded_input);
    std::vector<at::cuda::CUDAStream> streams;

    #pragma omp parallel for num_threads(batch_size)    
    for(int batch = 0; batch < batch_size; batch++) {
        auto stream = at::cuda::getStreamFromPool(false);
        auto stream_t = cudaStream_t(stream);
        auto img = padded_input[batch];
        auto mask = masks[batch];
        AT_DISPATCH_FLOATING_TYPES(img.type(), "ampere_cuda_kernel", ([&] {
        ampere_cuda_kernel<scalar_t><<<grid_size, block_size, 0, stream_t>>>(
            img.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
            );
        }));
    }
    
    torch::cuda::synchronize();
    masks = masks.index({Slice(), Slice(None, h), Slice(None, w)});
    return masks;
}