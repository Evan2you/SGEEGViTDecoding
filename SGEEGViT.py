import torch
import torch.nn as nn
import torch.nn.functional as F

class SAFCModule(nn.Module):
    def __init__(self, chans, d_model):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, d_model, chans, 1))
        self.bias = nn.Parameter(torch.zeros(1, d_model, chans, 1))
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, (chans, 1), groups=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        g = torch.mean(x, dim=(2, 3), keepdim=True) 
        gate = torch.sigmoid(self.gain * g + self.bias)
        x_calibrated = x * gate
        return self.spatial_conv(x_calibrated)

class GECNBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.subband_weights = nn.Parameter(torch.ones(4)) 
        self.gcn_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_spectral):
        weights = F.softmax(self.subband_weights, dim=0)
        h0 = torch.sum(x_spectral * weights.view(1, 4, 1, 1), dim=1)
        return self.gcn_conv(h0.unsqueeze(-1))

class EVTFFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
    def forward(self, f_spatial, f_spectral):
        Q = self.query_proj(f_spectral)
        K = self.key_proj(f_spatial)
        V = self.value_proj(f_spatial)
        
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / (Q.shape[-1]**0.5), dim=-1)
        out = torch.bmm(attn, V)
        return out + f_spectral

class SGEEGVIT(nn.Module):
    def __init__(self, num_classes=6, chans=11, d_model=32):
        super().__init__()
        self.init_embed = nn.Conv2d(1, d_model, (1, 1))
        
        self.safc = SAFCModule(chans, d_model)
        self.gecn = GECNBranch(d_model=d_model)
        self.fusion = EVTFFusion(d_model)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x_raw, x_spectral_subbands):
        x = self.init_embed(x_raw)
        feat_spatial = self.safc(x)
        
        feat_graph = self.gecn(x_spectral_subbands)
        
        s_feat = feat_graph.flatten(2).transpose(1, 2)
        x_feat = feat_spatial.flatten(2).transpose(1, 2)
        fused = self.fusion(x_feat, s_feat)
        
        return self.classifier(fused)

if __name__ == "__main__":
    model = SGEEGVIT(num_classes=6, chans=11, d_model=16) 
    raw_data = torch.randn(1, 1, 11, 1000)
    spectral_data = torch.randn(1, 4, 1, 16)
    
    output = model(raw_data, spectral_data)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e3:.2f} K")
    print(f"Output Shape: {output.shape}")