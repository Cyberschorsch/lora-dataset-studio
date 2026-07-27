"""Z-Image Turbo — the THIRD local generation engine, next to Klein and Krea.

WHAT IT IS
----------
Z-Image Turbo (Tongyi-MAI, Apache-2.0) is a fast distilled base text-to-image
model. It has NO identity-edit mechanism of its own (unlike Krea's edit LoRA +
custom node pack). So this engine generates via IMG2IMG: the dataset reference is
VAE-encoded and fed to the sampler as the init latent at an adjustable denoise.
Identity is therefore LOOSE — composition and coloring carry over, not a faithful
face — and the UI states this honestly. It is the ONLY reference mechanism that a
plain base model supports with core ComfyUI nodes.

THE GRAPH (core nodes only — no custom pack, unlike Krea)
--------------------------------------------------------
    UNETLoader ─┐                        BasicScheduler(denoise) ─┐
    CLIPLoader(type='lumina2') ─ CLIPTextEncode(±) ─ CFGGuider ─┐ │
    VAELoader ─ VAEEncode(reference) ── latent ─────────────────┼─┼─ SamplerCustomAdvanced ─ VAEDecode ─ SaveImage
    LoadImage(reference) ──────────────────────────────────────┘ │  ▲ KSamplerSelect(euler)  ▲ RandomNoise(seed)

Assets (3, no LoRA): UNET + text-encoder + VAE. Because it is core-nodes-only,
there is NO custom-node preflight (like Klein, unlike Krea).
"""
from __future__ import annotations
import logging
import os

from .. import config as cfg
from . import comfy_model_paths

logger = logging.getLogger(__name__)

ENGINE_ID = 'zimage'
ENGINE_LABEL = 'Z-Image Turbo'

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')

# Display paths for the "place it here" Setup message. Real lookup goes through
# comfy_model_paths, so extra_model_paths.yaml roots work identically.
ZIMAGE_ASSETS = {
    'zimage_model': {
        'kind': 'Z-Image Turbo base model',
        'path': 'models/diffusion_models/z image/z_image_turbo_bf16.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/z_image_turbo',
    },
    'zimage_text_encoder': {
        'kind': 'Qwen3-4B text encoder',
        'path': 'models/text_encoders/qwen_3_4b.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/z_image_turbo',
    },
    'zimage_vae': {
        'kind': 'Z-Image VAE',
        'path': 'models/vae/z_image_ae.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/z_image_turbo',
    },
}
ZIMAGE_REQUIRED = tuple(ZIMAGE_ASSETS)


def _listings(comfy_type):
    out = []
    for folder in comfy_model_paths.search_roots(comfy_type):
        try:
            out.append((folder, sorted(n for n in os.listdir(folder)
                                       if n.lower().endswith(_MODEL_SUFFIXES))))
        except OSError:
            continue
    return out


def _find_model_file(comfy_type, canonical, tokens):
    """Canonical name if present in ANY search root, else the first (sorted) name
    containing a NARROW token. None when nothing matches — never a blind guess."""
    listings = _listings(comfy_type)
    if any(canonical in names for _root, names in listings):
        return canonical
    for _root, names in listings:
        for n in names:
            if any(tok in n.lower() for tok in tokens):
                return n
    return None


def _zimage_unet_folders():
    """(prefix, [model files]) for the Z-Image UNET across every diffusion-model
    search root. `prefix` is a 'z image'/'zimage'/'z-image' subfolder or '' for a
    file dropped straight into a search root. Mirrors krea_edit_helper."""
    out = []
    for base_dir in comfy_model_paths.search_roots('diffusion_models'):
        try:
            entries = os.listdir(base_dir)
        except OSError:
            continue
        subs = sorted(d for d in entries
                      if _is_zimage_token(d) and os.path.isdir(os.path.join(base_dir, d)))
        for sub in subs:
            try:
                names = sorted(n for n in os.listdir(os.path.join(base_dir, sub))
                               if n.lower().endswith(_MODEL_SUFFIXES))
            except OSError:
                continue
            if names:
                out.append((sub, names))
        root_names = sorted(n for n in entries
                            if _is_zimage_token(n) and n.lower().endswith(_MODEL_SUFFIXES)
                            and os.path.isfile(os.path.join(base_dir, n)))
        if root_names:
            out.append(('', root_names))
    return out


def _is_zimage_token(name):
    low = (name or '').lower()
    return 'z image' in low or 'zimage' in low or 'z-image' in low


def resolve_zimage_unet(selected=None):
    """ComfyUI-relative `unet_name` WITH its subfolder prefix (e.g.
    'z image\\z_image_turbo_bf16.safetensors'), or None when no Z-Image UNET is on
    disk. Preference: the explicit pick (`selected` or the `zimage.base_model`
    setting, matched on BASENAME), then a 'turbo' build, then the first candidate.
    Deterministic — the same install always resolves the same file."""
    folders = _zimage_unet_folders()
    if not folders:
        return None
    pick = selected or cfg.get('zimage.base_model') or ''
    bare_pick = os.path.basename(str(pick).replace('/', os.sep).replace('\\', os.sep))
    if bare_pick:
        for sub, names in folders:
            if bare_pick in names:
                return os.path.join(sub, bare_pick)
        logger.warning('zimage.base_model %r not found under any z-image folder — '
                       'falling back to automatic resolution', pick)
    for sub, names in folders:
        for n in names:
            if 'turbo' in n.lower():
                return os.path.join(sub, n)
    sub, names = folders[0]
    return os.path.join(sub, names[0])


def resolve_zimage_text_encoder():
    """`clip_name` for the CLIPLoader (type='lumina2'). Canonical qwen_3_4b, else a
    NARROW qwen-3-4b token. Never qwen_3_8b (Klein) or qwen3vl_4b (Krea)."""
    return _find_model_file('text_encoders', 'qwen_3_4b.safetensors',
                            ('qwen_3_4b', 'qwen3_4b', 'qwen3-4b', 'qwen_3-4b'))


def resolve_zimage_vae():
    """`vae_name` for the VAELoader. Canonical downloaded name z_image_ae, else the
    user's manual 'z ae.safetensors' / z-image-ae tokens. NEVER bare 'ae' or
    flux2-vae — Flux's VAE is also named ae.safetensors and must not be picked."""
    return _find_model_file('vae', 'z_image_ae.safetensors',
                            ('z_image_ae', 'z ae', 'z-ae', 'zimage_ae', 'z_ae'))


def zimage_missing_assets():
    """Which Z-Image assets are NOT on disk, as ZIMAGE_ASSETS keys. Disk-only,
    network-free — safe for the readiness probe."""
    missing = []
    if not resolve_zimage_unet():
        missing.append('zimage_model')
    if not resolve_zimage_text_encoder():
        missing.append('zimage_text_encoder')
    if not resolve_zimage_vae():
        missing.append('zimage_vae')
    return missing
