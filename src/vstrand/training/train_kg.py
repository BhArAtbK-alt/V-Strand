"""Train the neural KG head on synthetic planted triples; reports loss with/without ontology."""
import torch, torch.nn.functional as F
from ..config import load_config
from ..kg.predicate_mlp import PredicateKG
from ..kg.ontology import ontology_loss

def synth_batch(B, E, vd, nu, nb, seed=0):
    g = torch.Generator().manual_seed(seed)
    video = torch.randn(B, vd, generator=g)
    w_u = torch.randn(vd, nu, generator=g); w_b = torch.randn(vd, nb, generator=g)
    unary = (torch.sigmoid(video @ w_u)[:, None, :].expand(B, E, nu) > 0.5).float()
    binary = (torch.sigmoid(video @ w_b)[:, None, None, :].expand(B, E, E, nb) > 0.5).float()
    binary[..., 1] = torch.maximum(binary[..., 1], binary[..., 0])      # ground truth obeys fold=>change
    return video, unary, binary

def train(cfg=None, use_ontology=True, epochs=None, verbose=True):
    k = (cfg or load_config())["kg"]; epochs = epochs or k["epochs"]
    torch.manual_seed(0)
    model = PredicateKG(8, k["entity_dim"], k["video_dim"], k["n_unary"], k["n_binary"])
    opt = torch.optim.Adam(model.parameters(), lr=k["lr"])
    video, yu, yb = synth_batch(256, 8, k["video_dim"], k["n_unary"], k["n_binary"])
    for ep in range(epochs):
        pu, pb = model(video)
        sup = F.binary_cross_entropy(pu, yu) + F.binary_cross_entropy(pb, yb)
        onto = ontology_loss(pb)
        loss = sup + (k["ontology_weight"] * onto if use_ontology else 0)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % 15 == 0:
            print(f"[kg] ep {ep:3d} supervised={sup.item():.3f} rule_violation={onto.item():.4f}")
    return model, sup.item(), onto.item()

if __name__ == "__main__":
    train()
