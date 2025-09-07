import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

class VGG19(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40]) # :53, 52 is last maxpool
        

    def forward(self, x, **kwargs):        
        feats = {}
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats[scale] = x
                scale = scale*2
            x = layer(x)
        return feats

class Encoder(nn.Module):
    def __init__(self, dinov3_weights = None):
        super().__init__()       
    
        if dinov3_weights is None:
            REPO_DIR = "src/models/"
            MODEL_PATH = "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=MODEL_PATH).cuda()
        else:
            dinov3 = dinov3_weights

        self.cnn = VGG19(pretrained=True)
        self.dinov3 = dinov3 

        self.dinov3.eval()
        for p in self.dinov3.parameters():
            p.requires_grad = False
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def extract_dino_feature(self, x):
        B,C,H,W = x.shape
        with torch.no_grad():
            dinov3_features_16 = self.dinov3.forward_features(x)
            features_16 = dinov3_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//16, W//16)
        return features_16
    
    def forward(self, x):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)                        
        with torch.no_grad():                
            dinov3_features_16 = self.dinov3.forward_features(x)
            features_16 = dinov3_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//16, W//16)
            del dinov3_features_16
            feature_pyramid[16] = features_16
            
        return feature_pyramid
    
if __name__ == "__main__":
    REPO_DIR = "src/models/"
    MODEL_PATH = "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    encoder = Encoder().cuda()

    dummy_input = torch.randn(1, 3, 512, 512).to(torch.float32).cuda()
    feature_pyramid = encoder(dummy_input)
    print("")
        
