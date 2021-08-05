import torch
import masking


a = torch.randn(8,8).cuda()
print(a)
mask_a = masking.ampere(a, False)
print(mask_a)

b = torch.randn(2, 2, 8, 8).cuda()
print(b)
b = b.view(-1, 8, 8)
mask_b = masking.ampere(b, False).view(2, 2, 8, 8)
print(mask_b)

c = torch.randn(2, 2, 8, 8).cuda()
print(c)
c = c.view(-1, 8, 8)
mask_c = masking.ampere(c, True).view(2,2,8,8)
print(mask_c)