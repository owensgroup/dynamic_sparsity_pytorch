import torch
import masking

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch import Tensor
from masking import ampere_mask_cuda as ampere_mask
from typing import Optional
class AmpereConv2D(nn.Conv2d):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: _size_2_t = 0,
    dilation: _size_2_t = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros'):
      super().__init__(in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
      
  def mask_caller(self, input):
      inp_shape = input.shape
      pad_h = 0
      pad_w = 0
      if input.size(2) % 4 != 0:
        pad_h = 4 - input.size(0) % 4
      if input.size(3) % 4 != 0:
        pad_w = 4 - input.size(1) % 4
      input = F.pad(input, (0,0,0,0,0,pad_h,0,pad_w), 'constant', 0)
      mask = ampere_mask(input)
      mask = mask[:, :, :inp_shape[2], :inp_shape[3]]
      return mask

  def forward(self, input: Tensor) -> Tensor:
      
      input = input.flatten()
      with torch.no_grad():
        mask = ampere_mask(input)
      input = mask * input
      input = input.view(s)
      return super().forward(input)