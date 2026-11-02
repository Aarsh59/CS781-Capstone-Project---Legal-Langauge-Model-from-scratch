"""YOUR FILE: QLoRA — Dettmers et al. (2023) [9].

WHAT IT IS
  LoRA on top of a QUANTISED frozen base: the frozen W0 of every adapted linear is stored in 4 bits and dequantised
  on the fly in the forward pass; only the LoRA A/B matrices (kept in 16/32-bit) train. The paper's pieces:
    * NF4 ("4-bit NormalFloat"): 16 fixed quantisation levels placed at quantiles of a standard normal, so normally
      distributed weights use all 16 levels evenly;
    * block-wise quantisation: weights split into blocks (e.g. 64 values); each block stores its absmax as the scale,
      values stored as the index (0..15) of the nearest NF4 level after dividing by absmax;
    * double quantisation (optional): the per-block absmax constants are themselves quantised (8-bit) to save memory;
    * paged optimizers (memory paging) — not needed here.
  Forward: W0_hat = dequant(codes, absmax);  h = x @ W0_hat^T + bias + scale * B(A(x)).

WHAT THE PLUMBING CALLS
  apply(backbone, **cfg) -> None, e.g. config {"name": "qlora", "r": ..., "alpha": ..., "targets": [...], "block_size": ...}

SUGGESTED STRUCTURE (from scratch, PyTorch only)
  NF4_LEVELS = torch.tensor([...16 values...])           # the paper / bitsandbytes NF4 table
  def quantize_nf4(w, block_size) -> (codes uint8, absmax float)   # per block: absmax, then nearest-level index
  def dequantize_nf4(codes, absmax, shape, block_size) -> tensor
  class QLoRALinear(nn.Module):
      __init__(self, base: nn.Linear, r, alpha, block_size):
          quantise base.weight once; register codes / absmax as BUFFERS (not parameters); drop the fp weight
          create LoRA A, B as in your lora.py (B = 0)
      forward(self, x): F.linear(x, dequantize(...).to(x.dtype), base.bias) + scaling * B(A(x))
  def apply(backbone, r, alpha, targets, block_size): freeze(backbone); replace_linears(... -> QLoRALinear ...)
  Note: the quantisation error of the frozen weights is part of the method; compare memory (peak_mem_GB in
  summary.json) with plain LoRA. (Using the bitsandbytes library instead would be a library implementation — if you
  do, cite it and declare it.)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from inject import freeze, unfreeze, replace_linears   # noqa: F401

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0
], dtype=torch.float32)
def quantize_nf4(w,block_size):
  assert w.numel() % block_size == 0
  blocks = w.reshape(-1,block_size)
  absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
  blocks = blocks/absmax[:, None]
  dist  = (blocks[..., None] - NF4_LEVELS.to(blocks.device)).abs()
  codes = dist.argmin(dim=-1).to(torch.uint8)
  return (codes,absmax)
def dequantize_nf4(codes,absmax,shape,block_size):
  output = NF4_LEVELS.to(codes.device)[codes.long()]
  output = output*absmax[:, None]
  return output.reshape(shape)
class QLoRALinear(nn.Module):
  def __init__(self,base:nn.Linear,r,alpha,block_size):
    super().__init__()
    self.r = r 
    self.alpha=alpha
    self.block_size=block_size
    codes, absmax = quantize_nf4(base.weight.detach(), block_size)
    self.register_buffer("codes", codes)
    self.register_buffer("absmax", absmax)
    self.in_features=base.in_features
    self.out_features=base.out_features
    self.shape = tuple(base.weight.shape)
    if base.bias is not None:
      self.register_buffer("bias", base.bias.detach().clone())
    else:
      self.bias = None
    self.A = nn.Parameter(torch.randn(r, self.in_features))
        
    self.A = nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    self.B = nn.Parameter(torch.zeros(self.out_features,r))
    self.scaling  = alpha/r
  def forward(self,x:torch.Tensor):
    base = F.linear(x, dequantize_nf4(self.codes,self.absmax,self.shape,self.block_size).to(x.dtype), self.bias)
    lora  = (x@self.A.T)@self.B.T
    return base+self.scaling*lora











def apply(backbone, r, alpha, targets,block_size):
    freeze(backbone)
    replaced = replace_linears(backbone,
                               lambda name, lin: any(target in name for target in targets),
                               lambda name, lin: QLoRALinear(lin, r, alpha,block_size))
