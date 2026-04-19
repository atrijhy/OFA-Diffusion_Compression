# networks_ofa_sd.py  —  OFA physical-slicing for Stable Diffusion v1.5 UNet
#
# Instead of subclassing every diffusers class (fragile across versions), we use
# a **hook-based** approach:
#
#   1. Load a standard UNet2DConditionModel (full width).
#   2. At each training step, sample a subnet_cfg from saved masks.
#   3. Call `apply_slice_hooks(unet, subnet_cfg)` to register forward hooks
#      that physically slice weights on the fly.
#   4. After forward+backward, call `remove_slice_hooks(unet)` to clean up.
#
# This keeps the model 100% compatible with diffusers checkpointing and avoids
# any version-dependent subclass issues.
#
# Slicing strategy per pipe (matches sd_prune_physical.py):
#
#   Conv pipe:  conv1 output rows[:conv_keep]  +  conv2 input cols[:conv_keep]
#               norm2[:conv_keep],  time_emb_proj output[:conv_keep]
#
#   Attn pipe:  For self-attn (attn1) and cross-attn (attn2):
#               to_q/to_k/to_v output rows — per-head-offset slicing
#               to_out.0 input cols — per-head-offset slicing
#               Also: norm1/norm2/norm3 (LayerNorm) sliced to attn_keep*nh
#               proj_in/proj_out output/input sliced to attn_keep*nh
#
#   FFN pipe:   GEGLU ff.net.0.proj output rows[:ffn_keep*2] (gate+value halves)
#               ff.net.2 input cols[:ffn_keep]
#
# The attn pipe slices the "inner_dim" of the transformer block uniformly.
# Since SD v1.5 always has heads=8 with varying head_dim, we slice head_dim
# while keeping heads=8 fixed.  attn_keep = kept channels per head.
#
# IMPORTANT: We do NOT slice the Transformer2DModel's proj_in/proj_out or its
# GroupNorm because they bridge between the ResNet's spatial feature map
# (out_channels) and the transformer's inner_dim.  In SD v1.5, these are always
# equal (e.g., both 320 for block 0).  The conv pipe already controls
# out_channels; proj_in/proj_out/GN are NOT part of the attn pipe — they stay
# at full width.  Only the layers INSIDE BasicTransformerBlock get sliced.
#
# Similarly, we do NOT slice skip connections (conv_shortcut) — they must
# match the full out_channels to maintain the UNet skip-connection protocol.

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Dict, Optional, Any, List

# ---------------------------------------------------------------------------
# Utility: compute per-head row/col indices from keep_idx (within-head offsets)
# ---------------------------------------------------------------------------

def _per_head_rows(keep_idx: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Build absolute row indices for Q/K/V linear layers.

    keep_idx: [kd] offsets within [0, head_dim) to keep.
    Returns:  [num_heads * kd] absolute indices.
    """
    rows = []
    for h in range(num_heads):
        rows.append(h * head_dim + keep_idx)
    return torch.cat(rows)


def _per_head_geglu_rows(keep_idx: torch.Tensor, ff_inner: int) -> torch.Tensor:
    """Build absolute row indices for GEGLU proj (output dim = 2 * ff_inner).

    GEGLU proj outputs [gate_half | value_half], each of size ff_inner.
    keep_idx: [fk] offsets within [0, ff_inner) to keep.
    Returns:  [2 * fk] absolute row indices.
    """
    return torch.cat([keep_idx, ff_inner + keep_idx])


# ---------------------------------------------------------------------------
# Sliced forward replacements
# ---------------------------------------------------------------------------

def _sliced_linear_rows(module: nn.Linear, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Linear forward keeping only `rows` output dimensions."""
    w = module.weight[rows]
    b = module.bias[rows] if module.bias is not None else None
    return F.linear(x, w, b)


def _sliced_linear_cols(module: nn.Linear, x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Linear forward keeping only `cols` input dimensions.
    x must already be sliced to [B, ..., len(cols)]."""
    w = module.weight[:, cols]
    b = module.bias
    return F.linear(x, w, b)


def _sliced_conv2d_rows(module: nn.Conv2d, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Conv2d forward keeping only `rows` output channels."""
    w = module.weight[rows]
    b = module.bias[rows] if module.bias is not None else None
    return F.conv2d(x, w, b, module.stride, module.padding, module.dilation, module.groups)


def _sliced_conv2d_cols(module: nn.Conv2d, x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Conv2d forward keeping only `cols` input channels.
    x must already be sliced to have len(cols) channels."""
    w = module.weight[:, cols]
    b = module.bias
    return F.conv2d(x, w, b, module.stride, module.padding, module.dilation, module.groups)


def _sliced_groupnorm(x: torch.Tensor, norm: nn.GroupNorm, keep_idx: torch.Tensor) -> torch.Tensor:
    """GroupNorm on sliced channels. Recomputes num_groups to stay valid."""
    C = keep_idx.numel()
    ng = norm.num_groups
    while C % ng != 0 and ng > 1:
        ng -= 1
    return F.group_norm(
        x, ng,
        weight=norm.weight[keep_idx],
        bias=norm.bias[keep_idx],
        eps=norm.eps,
    )


def _sliced_layernorm(x: torch.Tensor, norm: nn.LayerNorm, keep_idx: torch.Tensor) -> torch.Tensor:
    """LayerNorm on sliced last dimension."""
    C = keep_idx.numel()
    return F.layer_norm(
        x, (C,),
        weight=norm.weight[keep_idx],
        bias=norm.bias[keep_idx] if norm.bias is not None else None,
        eps=norm.eps,
    )


# ---------------------------------------------------------------------------
# Hook-based sliced forward for ResnetBlock2D
# ---------------------------------------------------------------------------

class _SlicedResnetHook:
    """Replaces ResnetBlock2D.forward with conv-pipe slicing."""

    def __init__(self, block, conv_keep_idx: torch.Tensor):
        self.block = block
        self.conv_keep_idx = conv_keep_idx  # [ck] indices within [0, out_channels)

    def __call__(self, module, args, kwargs=None):
        # ResnetBlock2D.forward(input_tensor, temb)
        if kwargs is None:
            kwargs = {}
        input_tensor = args[0]
        temb = args[1] if len(args) > 1 else kwargs.get('temb', None)

        keep = self.conv_keep_idx
        hidden_states = input_tensor

        # norm1 (full width — operates on in_channels, not sliced)
        hidden_states = module.norm1(hidden_states)
        hidden_states = module.nonlinearity(hidden_states)

        # upsample / downsample (full width)
        if module.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = module.upsample(input_tensor)
            hidden_states = module.upsample(hidden_states)
        elif module.downsample is not None:
            input_tensor = module.downsample(input_tensor)
            hidden_states = module.downsample(hidden_states)

        # conv1: slice output rows
        hidden_states = _sliced_conv2d_rows(module.conv1, hidden_states, keep)

        # time_emb_proj: slice output rows
        if module.time_emb_proj is not None:
            if not module.skip_time_act:
                temb = module.nonlinearity(temb)
            temb_proj = _sliced_linear_rows(module.time_emb_proj, temb, keep)
            temb_proj = temb_proj[:, :, None, None]
        else:
            temb_proj = None

        # Add temb
        if temb_proj is not None and module.time_embedding_norm == "default":
            hidden_states = hidden_states + temb_proj

        # norm2: slice to conv_keep channels
        hidden_states = _sliced_groupnorm(hidden_states, module.norm2, keep)

        # scale_shift
        if temb_proj is not None and module.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb_proj, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = module.nonlinearity(hidden_states)
        hidden_states = module.dropout(hidden_states)

        # conv2: slice input cols
        hidden_states = _sliced_conv2d_cols(module.conv2, hidden_states, keep)

        # shortcut
        if module.conv_shortcut is not None:
            input_tensor = module.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / module.output_scale_factor
        return output_tensor


# ---------------------------------------------------------------------------
# Hook-based sliced forward for BasicTransformerBlock
# ---------------------------------------------------------------------------

class _SlicedTransformerBlockHook:
    """Replaces BasicTransformerBlock.forward with attn+ffn pipe slicing."""

    def __init__(self, block, attn_keep_idx: torch.Tensor, ffn_keep_idx: torch.Tensor,
                 num_heads: int, head_dim: int, ff_inner: int):
        self.block = block
        self.attn_keep_idx = attn_keep_idx  # [ak] per-head offsets in [0, head_dim)
        self.ffn_keep_idx = ffn_keep_idx    # [fk] offsets in [0, ff_inner)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_inner = ff_inner

        # Pre-compute absolute indices
        self.attn_rows = _per_head_rows(attn_keep_idx, num_heads, head_dim)  # [nh*ak]
        self.geglu_rows = _per_head_geglu_rows(ffn_keep_idx, ff_inner)       # [2*fk]
        self.ak = attn_keep_idx.numel()
        self.fk = ffn_keep_idx.numel()

    def __call__(self, module, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        hidden_states = args[0]
        attention_mask = kwargs.get('attention_mask', None)
        encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
        encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
        timestep = kwargs.get('timestep', None)
        cross_attention_kwargs = kwargs.get('cross_attention_kwargs', None)
        class_labels = kwargs.get('class_labels', None)

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        nh = self.num_heads
        ak = self.ak  # kept channels per head
        attn_rows = self.attn_rows  # [nh*ak] absolute indices for Q/K/V rows
        fk = self.fk
        geglu_rows = self.geglu_rows
        ffn_cols = self.ffn_keep_idx  # [fk]

        # IMPORTANT: hidden_states has shape [B, seq, inner_dim] (full width).
        # The Transformer2DModel's proj_in already mapped from spatial features
        # to inner_dim.  LayerNorms operate on full inner_dim.
        # We only slice the *internal* projections (Q/K/V, FFN), then project
        # back to full inner_dim via to_out / ff.net.2.  This keeps residual
        # connections at full width.

        # ---- 1. Self-Attention ----
        # norm1: full width LayerNorm (no slicing)
        norm_hidden_states = module.norm1(hidden_states)

        # Sliced self-attention
        attn_output = self._sliced_attention(
            module.attn1, norm_hidden_states,
            encoder_hidden_states=None,
            attn_rows=attn_rows, nh=nh, ak=ak,
            is_cross=False,
        )
        hidden_states = attn_output + hidden_states

        # ---- 2. Cross-Attention ----
        if module.attn2 is not None:
            norm_hidden_states = module.norm2(hidden_states)

            attn_output = self._sliced_attention(
                module.attn2, norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attn_rows=attn_rows, nh=nh, ak=ak,
                is_cross=True,
            )
            hidden_states = attn_output + hidden_states

        # ---- 3. Feed-Forward ----
        norm_hidden_states = module.norm3(hidden_states)

        # GEGLU: ff.net[0].proj — input is full inner_dim, slice output rows
        geglu_mod = module.ff.net[0]  # GEGLU module
        geglu_out = _sliced_linear_rows(geglu_mod.proj, norm_hidden_states, geglu_rows)
        # Split gate and value, apply gelu
        value_half, gate_half = geglu_out.chunk(2, dim=-1)
        ff_hidden = value_half * F.gelu(gate_half)

        # Dropout
        ff_hidden = module.ff.net[1](ff_hidden)

        # ff.net[2] (Linear: ff_inner -> inner_dim)
        # Input: [B, seq, fk] (sliced), cols = ffn_cols
        # Output: [B, seq, inner_dim] (full width, all rows)
        ff_down = module.ff.net[2]
        w = ff_down.weight[:, ffn_cols]  # [inner_dim, fk]
        b = ff_down.bias               # [inner_dim]
        ff_output = F.linear(ff_hidden, w, b)

        hidden_states = ff_output + hidden_states

        return hidden_states

    def _sliced_attention(self, attn_mod, hidden_states, encoder_hidden_states,
                          attn_rows, nh, ak, is_cross):
        """Perform sliced attention (self or cross).

        Input hidden_states: [B, seq, inner_dim] (full width from LayerNorm).
        Q/K/V projections: from full inner_dim → sliced nh*ak.
        to_out: from sliced nh*ak → full inner_dim.
        """
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)
        else:
            B = hidden_states.shape[0]

        if is_cross and encoder_hidden_states is not None:
            kv_input = encoder_hidden_states
        else:
            kv_input = hidden_states

        # Q: from full-width hidden_states, slice output rows only
        query = _sliced_linear_rows(attn_mod.to_q, hidden_states, attn_rows)

        # K, V: input is full-width (either hidden_states or CLIP embeddings)
        key = _sliced_linear_rows(attn_mod.to_k, kv_input, attn_rows)
        value = _sliced_linear_rows(attn_mod.to_v, kv_input, attn_rows)

        # Reshape for multi-head: [B, seq, nh*ak] -> [B, nh, seq, ak]
        query = query.view(B, -1, nh, ak).transpose(1, 2)
        key = key.view(B, -1, nh, ak).transpose(1, 2)
        value = value.view(B, -1, nh, ak).transpose(1, 2)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        # [B, nh, seq, ak] -> [B, seq, nh*ak]
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, nh * ak)
        attn_output = attn_output.to(query.dtype)

        # to_out[0]: Linear(inner_dim, inner_dim)
        # Input cols = attn_rows (sliced), output = full inner_dim (all rows)
        to_out_linear = attn_mod.to_out[0]
        w = to_out_linear.weight[:, attn_rows]  # [inner_dim, nh*ak]
        b = to_out_linear.bias                  # [inner_dim]
        attn_output = F.linear(attn_output, w, b)

        # to_out[1]: Dropout
        attn_output = attn_mod.to_out[1](attn_output)

        return attn_output


# ---------------------------------------------------------------------------
# Main API: apply / remove hooks
# ---------------------------------------------------------------------------

_HOOK_HANDLES_KEY = '_ofa_slice_hooks'


def apply_slice_hooks(unet: nn.Module, subnet_cfg: Dict[str, Any]) -> None:
    """Register forward-pre-hooks on ResnetBlock2D and BasicTransformerBlock
    modules to perform physical slicing during forward pass.

    subnet_cfg format (from sd_prune_physical.py masks):
    {
        'layer_name': {                     # e.g. 'down_blocks.0.resnets.0'
            'conv_keep': int,               # number of conv channels to keep
            'attn_keep': int,               # channels per head to keep
            'ffn_keep': int,                # FFN inner dim to keep
            'conv_keep_idx': Tensor[ck],    # actual indices (sorted)
            'attn_keep_idx': Tensor[ak],    # per-head offsets
            'ffn_keep_idx': Tensor[fk],     # FFN inner offsets
        },
        ...
    }
    """
    remove_slice_hooks(unet)  # clean up any existing hooks
    handles = []
    named_mods = dict(unet.named_modules())

    for layer_name, cfg in subnet_cfg.items():
        # --- Conv pipe: ResnetBlock2D ---
        resnet_mod = named_mods.get(layer_name)
        if resnet_mod is None:
            continue

        conv_keep_idx = cfg.get('conv_keep_idx')
        if conv_keep_idx is not None:
            conv_keep_idx = conv_keep_idx.to(resnet_mod.conv1.weight.device)
            hook = _SlicedResnetHook(resnet_mod, conv_keep_idx)
            h = resnet_mod.register_forward_pre_hook(
                lambda mod, args, _hook=hook: (_hook(mod, args),),
                with_kwargs=False
            )
            # Actually, forward_pre_hook can't replace output. Use forward_hook instead.
            # Let's use a different approach: monkey-patch forward temporarily.
            # Remove the hook we just registered.
            h.remove()

            # Save original forward and monkey-patch
            if not hasattr(resnet_mod, '_ofa_orig_forward'):
                resnet_mod._ofa_orig_forward = resnet_mod.forward
            resnet_mod.forward = lambda *a, _hook=hook, _mod=resnet_mod, **kw: _hook(_mod, a, kw)
            handles.append(('monkey', resnet_mod))

        # --- Attn + FFN pipe: BasicTransformerBlock ---
        attn_keep_idx = cfg.get('attn_keep_idx')
        ffn_keep_idx = cfg.get('ffn_keep_idx')
        if attn_keep_idx is None and ffn_keep_idx is None:
            continue

        # Find the corresponding transformer block
        parts = layer_name.split('.')
        if 'resnets' not in parts:
            continue
        idx = parts.index('resnets')
        attn_parts = parts[:idx] + ['attentions'] + parts[idx+1:]
        attn_base = '.'.join(attn_parts)
        tb_path = f"{attn_base}.transformer_blocks.0"
        tb_mod = named_mods.get(tb_path)
        if tb_mod is None:
            continue

        # Get attention metadata
        attn1 = tb_mod.attn1
        num_heads = attn1.heads
        inner_dim = attn1.to_q.weight.shape[0]
        head_dim = inner_dim // num_heads

        # FFN metadata
        ff_proj = tb_mod.ff.net[0].proj  # GEGLU
        ff_inner = ff_proj.weight.shape[0] // 2  # GEGLU outputs 2*ff_inner

        device = attn1.to_q.weight.device

        if attn_keep_idx is None:
            attn_keep_idx = torch.arange(head_dim, device=device)
        else:
            attn_keep_idx = attn_keep_idx.to(device)

        if ffn_keep_idx is None:
            ffn_keep_idx = torch.arange(ff_inner, device=device)
        else:
            ffn_keep_idx = ffn_keep_idx.to(device)

        hook = _SlicedTransformerBlockHook(
            tb_mod, attn_keep_idx, ffn_keep_idx,
            num_heads=num_heads, head_dim=head_dim, ff_inner=ff_inner,
        )

        if not hasattr(tb_mod, '_ofa_orig_forward'):
            tb_mod._ofa_orig_forward = tb_mod.forward
        tb_mod.forward = lambda *a, _hook=hook, _mod=tb_mod, **kw: _hook(_mod, a, kw)
        handles.append(('monkey', tb_mod))

    # Store handles for cleanup
    unet._ofa_slice_handles = handles


def remove_slice_hooks(unet: nn.Module) -> None:
    """Remove all OFA slice hooks and restore original forwards."""
    handles = getattr(unet, '_ofa_slice_handles', [])
    for typ, obj in handles:
        if typ == 'monkey' and hasattr(obj, '_ofa_orig_forward'):
            obj.forward = obj._ofa_orig_forward
            del obj._ofa_orig_forward
    unet._ofa_slice_handles = []


@contextmanager
def sliced_forward(unet: nn.Module, subnet_cfg: Dict[str, Any]):
    """Context manager: temporarily apply slicing hooks for one forward pass."""
    apply_slice_hooks(unet, subnet_cfg)
    try:
        yield unet
    finally:
        remove_slice_hooks(unet)


# ---------------------------------------------------------------------------
# Build subnet_cfg from mask file (produced by sd_prune_physical.py)
# ---------------------------------------------------------------------------

def build_subnet_cfg_from_masks(masks: Dict[str, Any], ratio: float = 1.0) -> Dict[str, Any]:
    """Convert mask file content to subnet_cfg dict for apply_slice_hooks.

    masks: the full dict saved by sd_prune_physical.py, i.e.
    {
        'masks': {P_i: {layer_name: {'conv_keep', 'attn_keep', 'ffn_keep'}}},
        'conv_internal_ranks': {layer_name: LongTensor[out_ch]},
        'attn_channel_ranks':  {layer_name: LongTensor[head_dim]},
        'ffn_internal_ranks':  {layer_name: LongTensor[ff_inner]},
    }

    ratio: 0.0 → smallest subnet (P_min), 1.0 → full network (P=1.0).

    The function maps `ratio` to the nearest pre-computed P_i and uses that
    P_i's importance-weighted keep counts together with the rank tensors to
    select which channels to keep.  This is more accurate than linear
    interpolation because the keep counts were derived from Formula 5
    (importance-proportional budget allocation per resolution group).
    """
    ratio = float(max(0.0, min(1.0, ratio)))

    # ── Primary path: compact format from sd_prune_physical.py ───────────────
    if isinstance(masks, dict) and 'masks' in masks and isinstance(masks.get('masks'), dict):
        all_p_masks = masks['masks']
        if not all_p_masks:
            return {}

        conv_ranks = masks.get('conv_internal_ranks', {})
        attn_ranks = masks.get('attn_channel_ranks', {})
        ffn_ranks  = masks.get('ffn_internal_ranks',  {})

        p_values = sorted(all_p_masks.keys(), key=float)
        p_min = float(p_values[0])
        p_max = float(p_values[-1])   # should be 1.0

        # Map ratio ∈ [0,1] → target P_i ∈ [p_min, p_max], then snap to nearest
        target_p = p_min + ratio * (p_max - p_min)
        closest_p = min(p_values, key=lambda p: abs(float(p) - target_p))
        keep_cfg = all_p_masks[closest_p]

        subnet_cfg = {}
        for layer_name, cfg in keep_cfg.items():
            if not isinstance(cfg, dict):
                continue
            layer: Dict[str, Any] = {}

            if 'conv_keep' in cfg and layer_name in conv_ranks:
                ck   = int(cfg['conv_keep'])
                rank = conv_ranks[layer_name].to(dtype=torch.long)
                layer['conv_keep_idx'] = rank[:ck].sort()[0]
                layer['conv_keep']     = ck

            if 'attn_keep' in cfg and layer_name in attn_ranks:
                ak   = int(cfg['attn_keep'])
                rank = attn_ranks[layer_name].to(dtype=torch.long)
                layer['attn_keep_idx'] = rank[:ak].sort()[0]
                layer['attn_keep']     = ak

            if 'ffn_keep' in cfg and layer_name in ffn_ranks:
                fk   = int(cfg['ffn_keep'])
                rank = ffn_ranks[layer_name].to(dtype=torch.long)
                layer['ffn_keep_idx'] = rank[:fk].sort()[0]
                layer['ffn_keep']     = fk

            if layer:
                subnet_cfg[layer_name] = layer

        return subnet_cfg

    # ── Guard: compact masks passed without rank tensors ─────────────────────
    if isinstance(masks, dict) and all(isinstance(k, (float, int)) for k in masks.keys()):
        raise ValueError(
            "Received compact masks without rank tensors. "
            "Pass the full mask object saved by sd_prune_physical.py."
        )

    # ── Legacy path: flat format  {layer_name: {conv_internal_rank, ...}} ────
    subnet_cfg = {}
    for layer_name, m in masks.items():
        if not isinstance(m, dict):
            continue
        cfg: Dict[str, Any] = {}

        conv_rank = m.get('conv_internal_rank')
        if conv_rank is not None:
            conv_rank    = conv_rank.to(dtype=torch.long)
            full_ch      = conv_rank.numel()
            conv_keep    = m.get('conv_keep', full_ch)
            conv_keep    = int(conv_keep + (full_ch - conv_keep) * ratio)
            conv_keep    = max(32, (conv_keep // 32) * 32)
            conv_keep    = min(conv_keep, full_ch)
            cfg['conv_keep_idx'] = conv_rank[:conv_keep].sort()[0]
            cfg['conv_keep']     = conv_keep

        attn_rank = m.get('attn_channel_rank')
        if attn_rank is not None:
            attn_rank = attn_rank.to(dtype=torch.long)
            full_hd   = attn_rank.numel()
            attn_keep = m.get('attn_keep', full_hd)
            attn_keep = int(attn_keep + (full_hd - attn_keep) * ratio)
            attn_keep = max(1, min(attn_keep, full_hd))
            cfg['attn_keep_idx'] = attn_rank[:attn_keep].sort()[0]
            cfg['attn_keep']     = attn_keep

        ffn_rank = m.get('ffn_internal_rank')
        if ffn_rank is not None:
            ffn_rank = ffn_rank.to(dtype=torch.long)
            full_ff  = ffn_rank.numel()
            ffn_keep = m.get('ffn_keep', full_ff)
            ffn_keep = int(ffn_keep + (full_ff - ffn_keep) * ratio)
            ffn_keep = max(32, (ffn_keep // 32) * 32)
            ffn_keep = min(ffn_keep, full_ff)
            cfg['ffn_keep_idx'] = ffn_rank[:ffn_keep].sort()[0]
            cfg['ffn_keep']     = ffn_keep

        if cfg:
            subnet_cfg[layer_name] = cfg

    return subnet_cfg


def sample_random_subnet_cfg(masks: Dict[str, Any],
                             ratio_range: tuple = (0.5, 1.0)) -> Dict[str, Any]:
    """Sample a random subnet by uniformly picking a ratio in ratio_range."""
    import random
    ratio = random.uniform(*ratio_range)
    return build_subnet_cfg_from_masks(masks, ratio=ratio)


def get_smallest_subnet_cfg(masks: Dict[str, Any]) -> Dict[str, Any]:
    """Get the smallest (most pruned) subnet."""
    return build_subnet_cfg_from_masks(masks, ratio=0.0)


def get_full_subnet_cfg(masks: Dict[str, Any]) -> Dict[str, Any]:
    """Get the full (unpruned) subnet — equivalent to no slicing."""
    return build_subnet_cfg_from_masks(masks, ratio=1.0)


# ---------------------------------------------------------------------------
# Paper's training sampling: linearly-decreasing weights (Section 5)
# ---------------------------------------------------------------------------

def sample_weighted_subnet_cfg(masks: Dict[str, Any], m: float = 3.0) -> Dict[str, Any]:
    """Sample ONE subnet config per training step using the paper's strategy.

    Section 5 of the OFA-Diffusion paper: at each update step randomly sample
    ONE retention rate P_i, then train only that subnet.  The sampling
    distribution w_{P_i} linearly descends with i (smallest P_1 gets the most
    weight), with w_{P_1} = m * w_{P_N} (default m = 3 per the paper).

    This is preferred over the sandwich rule, which the paper explicitly shows
    leads to an overfitted full subnet and underfitted smaller ones.
    """
    import random as _random

    if not (isinstance(masks, dict) and 'masks' in masks
            and isinstance(masks.get('masks'), dict)):
        # Legacy format fallback: uniform sampling
        return build_subnet_cfg_from_masks(masks, ratio=_random.random())

    p_values = sorted(masks['masks'].keys(), key=float)
    N = len(p_values)

    if N == 1:
        return build_subnet_cfg_from_masks(masks, ratio=0.0)

    # Linearly descending weights.
    # idx=0  → P_1 (smallest), weight = m  (highest)
    # idx=N-1 → P_N (full),    weight = 1  (lowest)
    # w_idx = 1 + (m - 1) * (N - 1 - idx) / (N - 1)
    raw_w = [1.0 + (m - 1.0) * (N - 1 - idx) / (N - 1) for idx in range(N)]
    total_w = sum(raw_w)
    probs = [w / total_w for w in raw_w]

    # Weighted random pick
    r = _random.random()
    cumulative = 0.0
    chosen_idx = N - 1
    for idx, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            chosen_idx = idx
            break

    # Map chosen P_i to ratio in [0, 1] so build_subnet_cfg_from_masks snaps
    # to exactly that P_i (nearest-neighbour lookup inside the function).
    p_min = float(p_values[0])
    p_max = float(p_values[-1])
    chosen_p = float(p_values[chosen_idx])
    ratio = (chosen_p - p_min) / (p_max - p_min) if p_max > p_min else 1.0

    return build_subnet_cfg_from_masks(masks, ratio=ratio)


# ---------------------------------------------------------------------------
# OFA sandwich sampling (kept for reference / ablation; paper prefers
# sample_weighted_subnet_cfg for fine-tuning from a pre-trained DPM)
# ---------------------------------------------------------------------------

def sandwich_sample_cfgs(masks: Dict[str, Any],
                         n_random: int = 2,
                         ratio_range: tuple = (0.5, 1.0)) -> List[Dict[str, Any]]:
    """Sample subnet configs for OFA sandwich rule:
    [full, smallest, random_1, ..., random_n]

    NOTE: The paper (Section 5) shows this strategy is suboptimal for
    OFA compression from a pre-trained DPM.  Use sample_weighted_subnet_cfg
    for training instead.
    """
    cfgs = [
        get_full_subnet_cfg(masks),
        get_smallest_subnet_cfg(masks),
    ]
    for _ in range(n_random):
        cfgs.append(sample_random_subnet_cfg(masks, ratio_range))
    return cfgs
