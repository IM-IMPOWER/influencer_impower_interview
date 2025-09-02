import torch, clip

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _pre = clip.load("ViT-B/32", device=_device)
_model.eval()

def embed_text(text: str):
    import torch
    toks = clip.tokenize([text]).to(_device)
    with torch.no_grad():
        z = _model.encode_text(toks)
        z = z / z.norm(dim=-1, keepdim=True)
    return z.squeeze(0).cpu().numpy()  # shape (512,)
