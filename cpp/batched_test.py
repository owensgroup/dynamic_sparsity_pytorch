import torch
import masking

a = torch.randn(384, 224, 224).cuda()
import torch.cuda.profiler as profiler


import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()
with torch.autograd.profiler.emit_nvtx():
    m = masking.batched_ampere(a)
