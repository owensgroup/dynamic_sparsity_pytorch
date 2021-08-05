import torch
import masking

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch import Tensor
import masking
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
  

  def forward(self, input: Tensor) -> Tensor:
      orig_shape = input.shape
      input = input.view(-1, orig_shape[2], orig_shape[3])
      mask = masking.ampere(input, True)
      input = mask * input
      input = input.view(orig_shape)
      return super().forward(input)