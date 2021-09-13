import torch
import masking
import torch.cuda.profiler as profiler


import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()

a = torch.randn(128, 64, 224, 224).cuda().view(-1,224,224)

with torch.autograd.profiler.emit_nvtx():
    m = masking.ampere(a, False)
    