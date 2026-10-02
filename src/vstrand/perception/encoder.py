"""Visual encoder wrapper. Real: LanguageBind (image + video towers).
Mock: deterministic random features so shapes can be tested on a laptop."""
import torch, torch.nn as nn

class MockLanguageBind(nn.Module):
    def __init__(self, vis_dim=1024, tokens_per_frame=16):
        super().__init__()
        self.vis_dim, self.tpf = vis_dim, tokens_per_frame

    @torch.no_grad()
    def encode_video(self, frames):          # [B,T,C,H,W]
        B, T = frames.shape[:2]
        g = torch.Generator().manual_seed(0)
        return torch.randn(B, T * self.tpf, self.vis_dim, generator=g)

    @torch.no_grad()
    def encode_image(self, images):          # [B,C,H,W]
        g = torch.Generator().manual_seed(1)
        return torch.randn(images.shape[0], self.tpf, self.vis_dim, generator=g)

def load_languagebind():
    """TODO(real): install LanguageBind; load its Video and Image towers.
    Both towers share one text-aligned space -> that is the point of the pillar."""
    raise NotImplementedError("Plug in LanguageBind here (see docs/architecture.md).")
