"""Pillar 6 - Latency-aware edge->cloud offloading (ApproxBit-style).
EDGE:  3D-conv channel compressor -> sigmoid -> hard 1-bit threshold (straight-through gradient)
CLOUD: asymmetric 3D-conv decoder reconstructs features for the heavy model.
Payload = 1 bit per code element."""
import torch, torch.nn as nn

class _BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return (x > 0.5).float()
    @staticmethod
    def backward(ctx, g): return g                       # straight-through

class EdgeEncoder(nn.Module):                            # tiny -> runs on Jetson
    def __init__(self, in_ch=64, code_ch=4):
        super().__init__()
        self.net = nn.Conv3d(in_ch, code_ch, kernel_size=3, padding=1)
    def forward(self, x):
        return _BinarizeSTE.apply(torch.sigmoid(self.net(x)))

class CloudDecoder(nn.Module):                           # bigger -> runs in cloud
    def __init__(self, in_ch=64, code_ch=4):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(code_ch, 128, 3, padding=1), nn.ReLU(),
                                 nn.Conv3d(128, in_ch, 3, padding=1))
    def forward(self, c): return self.net(c)

class ApproxBit(nn.Module):
    def __init__(self, in_ch=64, code_ch=4):
        super().__init__()
        self.enc, self.dec = EdgeEncoder(in_ch, code_ch), CloudDecoder(in_ch, code_ch)
    def forward(self, x):
        code = self.enc(x)
        return self.dec(code), code

def compression_ratio(feat, code, float_bits=32):
    return (feat.numel() * float_bits) / code.numel()    # raw fp32 bits / 1-bit code bits

def pack_bits(code):                                     # what actually goes over the network
    import numpy as np
    return np.packbits(code.detach().cpu().numpy().astype("uint8").ravel())
