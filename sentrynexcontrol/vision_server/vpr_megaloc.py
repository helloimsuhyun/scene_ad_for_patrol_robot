import torch
import numpy as np
import cv2
import torchvision.transforms as T

class MegaLocWrapper:
    def __init__(self, device="cuda"):
        self.device = device

        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)

        self.model = self.model.to(device)
        self.model.eval()

        self.tfm = T.Compose([
            T.ToTensor(),
            T.Resize((518, 518)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_image(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.tfm(img).unsqueeze(0).to(self.device)

        feat = self.model(x)

        if isinstance(feat, dict):
            feat = feat.get("global", feat.get("descriptor", feat))

        feat = torch.nn.functional.normalize(feat, dim=1)

        return feat.squeeze(0)