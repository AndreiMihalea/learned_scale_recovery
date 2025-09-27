import torch
import torch.nn as nn


class PairAttentionPoseNet(nn.Module):
    def __init__(self, encoder, feature_dim=512, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.encoder = encoder  # ResNet encoder from depth_and_egomotion

        # Global average pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # attention across pairs
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # regress relative poses
        self.pose_head = nn.Linear(hidden_dim, 6)

    def forward(self, pairs):
        """
        pairs: [B, N, 2*C, H, W]  (image pairs)
        """
        B, N, _, H, W = pairs.shape
        pair_feats = []
        
        for i in range(N):
            # Get features from the encoder (ResNet returns a list of features)
            features = self.encoder(pairs[:, i])  # List of features at different scales
            # Use the last (highest level) features
            f = features[-1]  # [B, 512, H', W']
            
            # Global average pooling
            f = self.global_pool(f)  # [B, 512, 1, 1]
            f = f.view(B, -1)  # [B, 512]
            
            # Project to hidden dimension
            f = self.proj(f)  # [B, hidden_dim]
            pair_feats.append(f)

        pair_feats = torch.stack(pair_feats, dim=1)  # [B, N, hidden_dim]

        # For single pairs, we can skip attention and directly use the features
        if N == 1:
            # Direct pose regression for single pairs
            poses = self.pose_head(pair_feats.squeeze(1)) * 0.01  # [B, 6]
            return poses[:, :3].unsqueeze(1).unsqueeze(2), poses[:, 3:].unsqueeze(1).unsqueeze(2)
        else:
            # attention across pairs
            attn_out = self.attn_encoder(pair_feats)  # [B, N, hidden_dim]
            poses = self.pose_head(attn_out).unsqueeze(2) * 0.01  # [B, N, 1, 6]
            return poses[..., :3], poses[..., 3:]
