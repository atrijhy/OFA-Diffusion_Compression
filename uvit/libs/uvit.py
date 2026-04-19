import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .timm import trunc_normal_
import einops
import torch.utils.checkpoint

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


# ---------------------------------------------------------------------------
# SliceMlp: MLP with optional dynamic physical slicing of hidden neurons
# ---------------------------------------------------------------------------

class SliceMlp(nn.Module):
    """Two-layer MLP with optional dynamic physical slicing of hidden neurons.

    When keep_idx is None (full network), behavior is identical to the original
    timm.Mlp.  When keep_idx is a LongTensor of neuron indices, only those
    hidden neurons are computed via F.linear with sliced weight rows/columns.
    Gradients flow only to the kept neurons; unkept neurons receive zero grad.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, keep_idx=None):
        if keep_idx is None:
            # Full network — identical to original timm.Mlp
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
        # Physical slice: forward only through kept neurons
        # fc1: [hidden, in] → slice rows → [k, in]
        h = F.linear(x, self.fc1.weight[keep_idx], self.fc1.bias[keep_idx])
        h = self.act(h)
        h = self.drop(h)
        # fc2: [out, hidden] → slice columns → [out, k]
        x = F.linear(h, self.fc2.weight[:, keep_idx], self.fc2.bias)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# SliceAttention: multi-head attention with per-head-offset physical slicing
# ---------------------------------------------------------------------------

def _per_head_qkv_rows(keep_idx, num_heads, head_dim):
    """Build absolute row indices into qkv.weight [3*nh*hd, embed_dim].

    keep_idx: LongTensor [kd] — within-head offsets to keep (0..head_dim-1).
    Returns:  LongTensor [3*nh*kd] — rows for Q_h0..Q_h(nh-1), K_h0.., V_h0..
    """
    nh, hd, kd = num_heads, head_dim, keep_idx.numel()
    rows = []
    for section in range(3):            # Q, K, V
        for h in range(nh):
            rows.append(section * nh * hd + h * hd + keep_idx)
    return torch.cat(rows)


def _per_head_proj_cols(keep_idx, num_heads, head_dim):
    """Build absolute column indices into proj.weight [embed_dim, nh*hd].

    keep_idx: LongTensor [kd] — within-head offsets to keep.
    Returns:  LongTensor [nh*kd]
    """
    cols = []
    for h in range(num_heads):
        cols.append(h * head_dim + keep_idx)
    return torch.cat(cols)


class SliceAttention(nn.Module):
    """Multi-head self-attention with optional per-head-offset physical slicing.

    When blk_cfg is None or contains no 'attn_keep_idx', the full forward is
    executed (identical to the original Attention class).

    When blk_cfg contains:
        'attn_keep_idx'  : LongTensor [kd] — within-head offsets to keep

    every head keeps the SAME offsets.  QKV rows and proj columns are computed
    from keep_idx on the fly.  The output dimension is always embed_dim
    (unchanged), so LayerNorm and skip connections are unaffected.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = qk_scale or self.head_dim ** -0.5

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attn_compute(self, qkv_out, B, L, nh, kd):
        """Shared attention computation for both full and sliced paths.

        qkv_out : [B, L, 3 * nh * kd]
        returns  : [B, L, nh * kd]
        """
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(
                qkv_out, 'B L (K H D) -> K B H L D', K=3, H=nh).float()
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(
                qkv_out, 'B L (K H D) -> K B L H D', K=3, H=nh)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = xformers.ops.memory_efficient_attention(q, k, v)
            return einops.rearrange(x, 'B L H D -> B L (H D)', H=nh)
        else:  # math
            qkv = einops.rearrange(
                qkv_out, 'B L (K H D) -> K B H L D', K=3, H=nh)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = kd ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            return (attn @ v).transpose(1, 2).reshape(B, L, nh * kd)

    def forward(self, x, blk_cfg=None):
        B, L, C = x.shape

        if blk_cfg is None or 'attn_keep_idx' not in blk_cfg:
            # Full network — identical to original Attention.forward
            qkv_out = self.qkv(x)
            x = self._attn_compute(qkv_out, B, L, self.num_heads, self.head_dim)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Per-head-offset physical slice path
        keep_idx = blk_cfg['attn_keep_idx']   # LongTensor [kd], within-head offsets
        nh = self.num_heads
        hd = self.head_dim
        kd = keep_idx.numel()

        # Build row/col index tensors
        qkv_rows  = _per_head_qkv_rows(keep_idx, nh, hd)    # [3*nh*kd]
        proj_cols = _per_head_proj_cols(keep_idx, nh, hd)    # [nh*kd]

        # Sliced QKV projection: only rows for kept offsets across all heads
        b_qkv   = (self.qkv.bias[qkv_rows]
                   if self.qkv.bias is not None else None)
        qkv_out = F.linear(x, self.qkv.weight[qkv_rows], b_qkv)  # [B, L, 3*nh*kd]

        # Attention with nh heads, each of dim kd
        x = self._attn_compute(qkv_out, B, L, nh, kd)             # [B, L, nh*kd]

        # Sliced output projection: only input columns for kept offsets
        x = F.linear(x, self.proj.weight[:, proj_cols], self.proj.bias)  # [B, L, C]
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = SliceAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp   = SliceMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                               act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None, blk_cfg=None):
        if self.use_checkpoint:
            # blk_cfg captured via closure; checkpoint replays tensor args x, skip
            def _fn(x_, skip_):
                return self._forward(x_, skip_, blk_cfg)
            return torch.utils.checkpoint.checkpoint(_fn, x, skip)
        return self._forward(x, skip, blk_cfg)

    def _forward(self, x, skip=None, blk_cfg=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x), blk_cfg)
        ffn_idx = blk_cfg.get('ffn_keep_idx') if blk_cfg else None
        x = x + self.mlp(self.norm2(x), ffn_idx)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None, subnet_cfg=None):
        """Forward pass with optional per-block physical slicing.

        subnet_cfg : dict or None
            If None, the full model is run (equivalent to P_i = 1.0 with no pruning).
            If a dict, it maps block names (e.g. 'in_blocks.0', 'mid_block',
            'out_blocks.2') to per-block configs:
                {
                    'ffn_keep_idx'  : LongTensor,   # hidden neuron indices to keep
                    'attn_keep_idx' : LongTensor,   # per-head-offset indices to keep
                }
            Build this dict via build_subnet_cfgs() in uvit_train_ofa_physical.py.
        """
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for i, blk in enumerate(self.in_blocks):
            blk_cfg = subnet_cfg.get(f'in_blocks.{i}') if subnet_cfg else None
            x = blk(x, blk_cfg=blk_cfg)
            skips.append(x)

        blk_cfg = subnet_cfg.get('mid_block') if subnet_cfg else None
        x = self.mid_block(x, blk_cfg=blk_cfg)

        for i, blk in enumerate(self.out_blocks):
            blk_cfg = subnet_cfg.get(f'out_blocks.{i}') if subnet_cfg else None
            x = blk(x, skips.pop(), blk_cfg=blk_cfg)

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x
