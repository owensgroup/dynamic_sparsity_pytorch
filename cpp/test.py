import torch
from masking import ampere_mask_cuda

a = torch.randn(8,8).cuda()
print(a)
mask = ampere_mask_cuda(a)
print(mask)