"""Train edge/cloud binary codec to reconstruct feature maps from 1-bit codes."""
import torch, torch.nn.functional as F
from ..config import load_config
from ..edge.approxbit import ApproxBit, compression_ratio

def train(cfg=None, epochs=None, verbose=True):
    e = (cfg or load_config())["edge"]; epochs = epochs or e["epochs"]
    torch.manual_seed(0)
    # synthetic smooth feature maps [B,C,T,H,W] with low-rank structure (real: LVLM visual feature maps)
    basis = torch.randn(8, e["in_channels"])
    coef = torch.randn(16, 8, 4, 8, 8)
    feats = torch.einsum("bkthw,kc->bcthw", coef, basis)
    model = ApproxBit(e["in_channels"], e["code_channels"])
    opt = torch.optim.Adam(model.parameters(), lr=e["lr"])
    for ep in range(epochs):
        rec, code = model(feats)
        loss = F.mse_loss(rec, feats)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % 20 == 0: print(f"[approxbit] ep {ep:3d} mse={loss.item():.4f}")
    base = F.mse_loss(torch.zeros_like(feats), feats).item()
    return model, {"mse": loss.item(), "baseline_mse": base, "compression_x": compression_ratio(feats, code)}

if __name__ == "__main__":
    print(train()[1])
