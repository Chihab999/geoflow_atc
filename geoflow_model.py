"""
GeoFlow-PC v8 — Model Architecture (Self-Contained)
=====================================================
Extracted from the full training script for standalone inference use.
Contains only the model architecture, geometry ops, and configuration —
no training loop, dataset classes, or server-specific paths.

Architecture:
  1. Multi-scale DGCNN encoder -> per-point + global features
  2. Transformer decoder (AdaPoinTr-style) -> coarse seeds
  3. Folding refinement with cross-attention -> fine output
  4. Merge partial + fine -> n_complete points

Reference:
  - DGCNN: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds" (2019)
  - AdaPoinTr: Yu et al., "Adaptive Point Transformer" (2023)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# CONFIG (Architecture-relevant subset only)
# ─────────────────────────────────────────────────────────────
CFG = dict(
    n_partial    = 512,
    n_complete   = 2048,
    encoder_dims = [64, 128, 256, 512],
    k_nn         = 16,
    d_model      = 384,
    n_heads      = 6,
    n_enc_layers = 6,
    n_dec_layers = 6,
    n_queries    = 256,
    fold_ratio   = 4,
)


# ─────────────────────────────────────────────────────────────
# GEOMETRY OPS (pure PyTorch, no CUDA extensions needed)
# ─────────────────────────────────────────────────────────────

def fps(pts, n):
    """Farthest Point Sampling. pts: (B,N,3) -> idx: (B,n)"""
    B, N, _ = pts.shape
    device = pts.device
    idx = torch.zeros(B, n, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    current = torch.randint(0, N, (B,), device=device)
    for i in range(n):
        idx[:, i] = current
        c = pts[torch.arange(B), current].unsqueeze(1)
        d = ((pts - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        current = dist.argmax(1)
    return idx


def gather(pts, idx):
    """pts: (B,N,C)  idx: (B,M) -> (B,M,C)"""
    B, N, C = pts.shape
    M = idx.shape[1]
    idx_exp = idx.unsqueeze(-1).expand(B, M, C)
    return pts.gather(1, idx_exp)


def knn(pts, k):
    """K nearest neighbours (self). pts: (B,N,3) -> idx: (B,N,k)"""
    B, N, _ = pts.shape
    pp = (pts ** 2).sum(-1, keepdim=True)
    pq = torch.bmm(pts, pts.transpose(1, 2))
    dist = pp + pp.transpose(1, 2) - 2 * pq
    dist = dist.clamp(min=0)
    idx = dist.topk(k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


def get_edge_features(x, idx):
    """Build edge features for EdgeConv.
    x: (B, C, N), idx: (B, N, k) -> (B, 2C, N, k)
    """
    B, C, N = x.shape
    k = idx.shape[-1]
    x_t = x.permute(0, 2, 1)
    idx_flat = idx.reshape(B, -1)
    neigh = gather(x_t, idx_flat).reshape(B, N, k, C)
    neigh = neigh.permute(0, 3, 1, 2)
    x_rep = x.unsqueeze(-1).expand_as(neigh)
    return torch.cat([x_rep, neigh - x_rep], dim=1)


# ─────────────────────────────────────────────────────────────
# MULTI-SCALE DGCNN ENCODER
# ─────────────────────────────────────────────────────────────

class EdgeConvBlock(nn.Module):
    """Single EdgeConv block with residual connection."""
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.shortcut = nn.Conv2d(in_ch * 2, out_ch, 1, bias=False) \
                        if in_ch * 2 != out_ch else nn.Identity()

    def forward(self, x):
        idx = knn(x.permute(0, 2, 1), self.k)
        edge = get_edge_features(x, idx)
        out = self.conv(edge).max(-1)[0]
        skip = self.shortcut(edge).max(-1)[0]
        return out + skip


class MultiScaleDGCNN(nn.Module):
    """Multi-scale DGCNN encoder.
    
    Input:  (B, N, 3)
    Output: per_point (B, N, d_model), global (B, d_model)
    """
    def __init__(self, k=16, dims=None, d_model=384):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256, 512]
        self.k = k
        self.input_proj = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.ec1 = EdgeConvBlock(64, dims[0], k)
        self.ec2 = EdgeConvBlock(dims[0], dims[1], k)
        self.ec3 = EdgeConvBlock(dims[1], dims[2], k)
        self.ec4 = EdgeConvBlock(dims[2], dims[3], k)

        total_ch = sum(dims)
        self.per_point_mlp = nn.Sequential(
            nn.Conv1d(total_ch, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape
        x = xyz.permute(0, 2, 1)
        x = self.input_proj(x)

        f1 = self.ec1(x)
        f2 = self.ec2(f1)
        f3 = self.ec3(f2)
        f4 = self.ec4(f3)

        ms = torch.cat([f1, f2, f3, f4], dim=1)
        per_point = self.per_point_mlp(ms)
        per_point = per_point.permute(0, 2, 1)

        g_max = per_point.max(1)[0]
        g_avg = per_point.mean(1)
        g = self.global_mlp(torch.cat([g_max, g_avg], dim=-1))
        return per_point, g


# ─────────────────────────────────────────────────────────────
# TRANSFORMER DECODER (AdaPoinTr-style coarse generation)
# ─────────────────────────────────────────────────────────────

class PositionEmbedding(nn.Module):
    """Learned position embedding from 3D coordinates."""
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, xyz):
        return self.mlp(xyz)


class CrossAttnBlock(nn.Module):
    """Cross-attention block with self-attention, cross-attention, and FFN."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, q_pos=None, kv_pos=None):
        q2 = self.ln1(q)
        if q_pos is not None:
            q2 = q2 + q_pos
        q = q + self.drop(self.self_attn(q2, q2, q2)[0])
        q2 = self.ln2(q)
        kv2 = kv
        if kv_pos is not None:
            kv2 = kv2 + kv_pos
        if q_pos is not None:
            q2 = q2 + q_pos
        q = q + self.drop(self.cross_attn(q2, kv2, kv2)[0])
        q = q + self.drop(self.ff(self.ln3(q)))
        return q


class TransformerDecoder(nn.Module):
    """Generates coarse point seeds via cross-attention on encoded partial cloud.
    
    Returns:
        seeds: (B, n_queries, 3)
        seed_feat: (B, n_queries, d)
    """
    def __init__(self, n_queries, d_model, n_heads, n_layers):
        super().__init__()
        self.n_queries = n_queries
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.pos_embed = PositionEmbedding(d_model)
        self.layers = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, memory, src_xyz):
        B = memory.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        kv_pos = self.pos_embed(src_xyz)
        for layer in self.layers:
            q = layer(q, memory, kv_pos=kv_pos)
        q = self.ln_out(q)
        seeds = self.head(q)
        return seeds, q


# ─────────────────────────────────────────────────────────────
# FOLDING REFINEMENT (fine output)
# ─────────────────────────────────────────────────────────────

class FoldingRefinement(nn.Module):
    """Generates fine point cloud from coarse seeds via folding + cross-attention.
    
    Output: (B, n_seeds * fold_ratio, 3)
    """
    def __init__(self, d_model, fold_ratio, n_seeds):
        super().__init__()
        self.fold_ratio = fold_ratio
        self.n_seeds = n_seeds
        self.register_buffer("grid", self._make_grid(fold_ratio))
        self.cross_attn = nn.MultiheadAttention(d_model, 6, batch_first=True)
        self.ln_ca = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 2 + d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    @staticmethod
    def _make_grid(r):
        side = int(math.ceil(math.sqrt(r)))
        xs = torch.linspace(-0.05, 0.05, side)
        ys = torch.linspace(-0.05, 0.05, side)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1)
        grid = grid.reshape(-1, 2)[:r]
        return grid

    def forward(self, coarse, seed_feat, partial, part_feat, global_f):
        B, n_s, _ = coarse.shape
        r = self.fold_ratio
        sf_ca, _ = self.cross_attn(seed_feat, part_feat, part_feat)
        seed_feat = self.ln_ca(seed_feat + sf_ca)
        seed_feat_exp = seed_feat.unsqueeze(2).expand(B, n_s, r, -1)
        coarse_exp = coarse.unsqueeze(2).expand(B, n_s, r, 3)
        global_exp = global_f.unsqueeze(1).unsqueeze(1).expand(B, n_s, r, -1)
        grid = self.grid.unsqueeze(0).unsqueeze(0).expand(B, n_s, -1, -1)
        feat = torch.cat([seed_feat_exp, grid, global_exp], dim=-1)
        offset = self.mlp(feat)
        fine = coarse_exp + offset
        fine = fine.reshape(B, n_s * r, 3)
        return fine


# ─────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────

class GeoFlowV8(nn.Module):
    """GeoFlow-PC v8 — Full point cloud completion model.
    
    Input:  partial (B, N_partial, 3)
    Output: coarse  (B, n_queries, 3), fine (B, n_complete, 3)
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.n_partial = cfg["n_partial"]
        self.n_complete = cfg["n_complete"]
        n_q = cfg["n_queries"]
        ratio = cfg["fold_ratio"]

        self.encoder = MultiScaleDGCNN(
            k=cfg["k_nn"], dims=cfg["encoder_dims"], d_model=d,
        )
        self.decoder = TransformerDecoder(
            n_queries=n_q, d_model=d,
            n_heads=cfg["n_heads"], n_layers=cfg["n_dec_layers"],
        )
        self.refiner = FoldingRefinement(
            d_model=d, fold_ratio=ratio, n_seeds=n_q,
        )
        self.seed_pos_embed = PositionEmbedding(d)

    def forward(self, partial):
        B = partial.shape[0]
        per_point, global_f = self.encoder(partial)
        coarse, seed_feat = self.decoder(per_point, partial)
        fine = self.refiner(coarse, seed_feat, partial, per_point, global_f)
        n_gen_needed = self.n_complete - self.n_partial
        if fine.shape[1] >= n_gen_needed:
            idx = fps(fine, n_gen_needed)
            gen = gather(fine, idx)
        else:
            rep = math.ceil(n_gen_needed / fine.shape[1])
            gen = fine.repeat(1, rep, 1)[:, :n_gen_needed, :]
        final = torch.cat([partial, gen], dim=1)
        return coarse, final
