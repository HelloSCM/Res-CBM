import clip
import torch
import torch.nn as nn


class CLIPFeaturizer(nn.Module):
    def __init__(self, backbone):
        super(CLIPFeaturizer, self).__init__()
        self.clip_featurizer, self.preprocess = clip.load(backbone, device='cpu')
        self.clip_featurizer.eval()
    
    def forward(self, img):
        with torch.no_grad():
            fea = self.clip_featurizer.encode_image(img)
        return fea