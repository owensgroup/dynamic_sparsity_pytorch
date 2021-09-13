import torch
import masking
import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

a = torch.randn(128, 64, 30, 30).cuda()
a = a.view(-1, 30, 30)
with torch.autograd.profiler.emit_nvtx():
    m = masking.ampere(a, True)