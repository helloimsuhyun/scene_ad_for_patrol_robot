# clip_embed.py
import numpy as np
import torch
import torch.nn.functional as F
import open_clip

# text > clip vector
class CLIPTextEmbedder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad() #output 
    def __call__(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)      # (1, T)
        feat = self.model.encode_text(tokens)                # (1, D)
        feat = F.normalize(feat, dim=-1)                     # cosine-ready
        return feat[0].float().cpu().numpy()                 # (D,)


"""
ex)
embedder = CLIPTextEmbedder()

v1 = embedder("a door is open")
v2 = embedder("the door is closed")

sim = v1 @ v2   # cosine similarity

"""