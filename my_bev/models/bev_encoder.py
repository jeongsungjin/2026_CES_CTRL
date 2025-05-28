import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSelfAttention(nn.Module):
    """시간적 self-attention 모듈"""
    def __init__(self, embed_dims=256):
        super().__init__()
        self.embed_dims = embed_dims
        self.qkv_proj = nn.Linear(embed_dims, embed_dims * 3)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        
    def forward(self, x, prev_x=None):
        # x: [B, H*W, C]
        B, N, C = x.shape
        
        # 이전 프레임이 있으면 concatenate
        if prev_x is not None:
            x = torch.cat([prev_x, x], dim=1)  # [B, 2*H*W, C]
        
        # QKV 프로젝션
        qkv = self.qkv_proj(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # [3, B, N, C]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention 계산
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)  # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # Output 프로젝션
        x = attn @ v  # [B, N, C]
        x = self.out_proj(x)
        
        # 현재 프레임 feature만 반환
        if prev_x is not None:
            x = x[:, -N:, :]
        return x

class SpatialCrossAttention(nn.Module):
    """공간적 cross-attention 모듈"""
    def __init__(self, embed_dims=256):
        super().__init__()
        self.embed_dims = embed_dims
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)  # kv_proj 대신 별도의 k, v projection
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        
        # 참조점 생성을 위한 convolution
        self.ref_points = nn.Conv2d(embed_dims, 2, 1)  # [x, y] 좌표
        
    def forward(self, query, key=None):
        # query: [B, H*W, C] - BEV 특징
        # key: [B, H*W, C] - 백본 특징 (없으면 self-attention)
        B, N, C = query.shape
        H = W = int(N ** 0.5)
        
        if key is None:
            key = query
            
        # 참조점 생성
        query_2d = query.transpose(1, 2).reshape(B, C, H, W)
        ref_points = self.ref_points(query_2d)  # [B, 2, H, W]
        ref_points = ref_points.permute(0, 2, 3, 1).reshape(B, N, 2)
        
        # QKV 프로젝션
        q = self.q_proj(query)  # [B, N, C]
        k = self.k_proj(key)    # [B, N, C]
        v = self.v_proj(key)    # [B, N, C]
        
        # 참조점 기반 attention mask 생성
        rel_pos = ref_points.unsqueeze(2) - ref_points.unsqueeze(1)  # [B, N, N, 2]
        dist = torch.norm(rel_pos, dim=-1)  # [B, N, N]
        mask = (dist < 0.1).float()  # 가까운 점들만 attention
        
        # Attention 계산
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)  # [B, N, N]
        attn = attn * mask
        attn = F.softmax(attn, dim=-1)
        
        # Output 프로젝션
        x = attn @ v  # [B, N, C]
        x = self.out_proj(x)
        return x

class BEVEncoder(nn.Module):
    """BEV 특징을 시공간적으로 강화하는 인코더"""
    def __init__(self, embed_dims=256, num_layers=3):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        
        # Encoder layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'temporal_attn': TemporalSelfAttention(embed_dims),
                'spatial_attn': SpatialCrossAttention(embed_dims),
                'norm1': nn.LayerNorm(embed_dims),
                'norm2': nn.LayerNorm(embed_dims),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dims, embed_dims * 4),
                    nn.ReLU(True),
                    nn.Linear(embed_dims * 4, embed_dims)
                ),
                'norm3': nn.LayerNorm(embed_dims)
            }) for _ in range(num_layers)
        ])
        
    def forward(self, bev_features, prev_bev=None, backbone_features=None):
        """
        Args:
            bev_features (Tensor): [B, C, H, W] 현재 프레임의 BEV 특징
            prev_bev (Tensor): [B, C, H, W] 이전 프레임의 BEV 특징 (옵션)
            backbone_features (Tensor): [B, C, H, W] 백본 특징 (옵션)
        """
        B, C, H, W = bev_features.shape
        
        # 2D → 시퀀스 변환
        x = bev_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        if prev_bev is not None:
            prev_x = prev_bev.flatten(2).transpose(1, 2)
        else:
            prev_x = None
            
        if backbone_features is not None:
            key = backbone_features.flatten(2).transpose(1, 2)
        else:
            key = None
        
        # Encoder layers
        for layer in self.layers:
            # Temporal self-attention
            residual = x
            x = layer['temporal_attn'](x, prev_x)
            x = residual + x
            x = layer['norm1'](x)
            
            # Spatial cross-attention
            residual = x
            x = layer['spatial_attn'](x, key)
            x = residual + x
            x = layer['norm2'](x)
            
            # FFN
            residual = x
            x = layer['ffn'](x)
            x = residual + x
            x = layer['norm3'](x)
        
        # 시퀀스 → 2D 변환
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x 