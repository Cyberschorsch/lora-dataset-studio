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

import random
import shutil
import uuid

from ..job_queue import queue_manager

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



class ZImageModelsMissing(Exception):
    """A Z-Image asset (base model / text encoder / VAE) is not on disk, so no
    valid job can be built. Raised BEFORE any row or job is created so the route
    answers ONE actionable 409. `.missing` = asset keys (subset of ZIMAGE_REQUIRED).
    Core-nodes-only engine, so there is no missing-nodes list."""

    def __init__(self, missing):
        self.missing = list(missing or [])
        super().__init__('Z-Image assets missing: ' + ', '.join(self.missing))


# Advisory floors, deliberately far under the real sizes so a legitimate file
# can never trip them; the structural cases (HTML page, truncation) need no floor.
ZIMAGE_MIN_BYTES = {
    'zimage_model': 1024 ** 3,               # 1 GB   (real bf16 ≈ 12 GB)
    'zimage_text_encoder': 256 * 1024 ** 2,  # 256 MB (real ≈ 8 GB)
    'zimage_vae': 8 * 1024 ** 2,             # 8 MB   (real ≈ 335 MB)
}


def _abs_under_roots(comfy_type, rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(comfy_type):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _zimage_asset_paths():
    """{ZIMAGE_ASSETS key: absolute path} for each asset PRESENT on disk."""
    paths = {}
    for key, comfy_type, rel in (
            ('zimage_model', 'diffusion_models', resolve_zimage_unet()),
            ('zimage_text_encoder', 'text_encoders', resolve_zimage_text_encoder()),
            ('zimage_vae', 'vae', resolve_zimage_vae())):
        p = _abs_under_roots(comfy_type, rel)
        if p:
            paths[key] = p
    return paths


def zimage_invalid_assets():
    """Z-Image assets on disk under the resolved name but NOT real weights (HTML
    gate page, truncated, tiny stub). Same [{asset, filename, verdict, blocking,
    reason}] shape as klein_invalid_assets, so one banner covers all engines."""
    from . import model_integrity
    out = []
    for asset, path in _zimage_asset_paths().items():
        res = model_integrity.validate_model_file(path, min_bytes=ZIMAGE_MIN_BYTES.get(asset))
        if res['ok']:
            continue
        out.append({'asset': asset, 'filename': res['filename'],
                    'verdict': res['verdict'], 'blocking': res['blocking'],
                    'reason': res['reason']})
    return out


def preflight():
    """Raise ZImageModelsMissing when the engine cannot run (any required asset
    absent). Present-but-invalid is surfaced separately by the readiness probe."""
    missing = zimage_missing_assets()
    if missing:
        raise ZImageModelsMissing(missing)



def build_workflow(source_image, prompt, *, unet, clip, vae, seed,
                   steps=8, denoise=0.75, filename_prefix='zimage_edit'):
    """ComfyUI API-format img2img graph, built from ZImage_bigLove_ZT3_optimal.json
    (the advanced-sampler form Z-Image uses) plus a LoadImage->VAEEncode init
    latent. Pure function of its arguments — no cfg read, no disk — so a test can
    assert the exact wiring without a ComfyUI. cfg is pinned to 1.0: Z-Image Turbo
    is guidance-distilled and ignores anything else. The reference is pre-sized by
    the caller (enqueue), so no scale node is needed here."""
    steps = 8 if steps is None else max(1, int(steps))
    denoise = 0.75 if denoise is None else float(denoise)
    return {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': unet, 'weight_dtype': 'default'},
              '_meta': {'title': 'Z-Image Turbo base model'}},
        '2': {'class_type': 'CLIPLoader',
              'inputs': {'clip_name': clip, 'type': 'lumina2'},
              '_meta': {'title': 'Qwen3-4B text encoder'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': vae}},
        '4': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': prompt, 'clip': ['2', 0]},
              '_meta': {'title': 'Positive'}},
        '5': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': '', 'clip': ['2', 0]},
              '_meta': {'title': 'Negative (empty)'}},
        '6': {'class_type': 'LoadImage', 'inputs': {'image': source_image}},
        '7': {'class_type': 'VAEEncode',
              'inputs': {'pixels': ['6', 0], 'vae': ['3', 0]},
              '_meta': {'title': 'Init latent (img2img)'}},
        '8': {'class_type': 'BasicScheduler',
              'inputs': {'scheduler': 'simple', 'steps': steps,
                         'denoise': denoise, 'model': ['1', 0]}},
        '9': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
        '10': {'class_type': 'CFGGuider',
               'inputs': {'cfg': 1.0, 'model': ['1', 0],
                          'positive': ['4', 0], 'negative': ['5', 0]}},
        '11': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed}},
        '12': {'class_type': 'SamplerCustomAdvanced',
               'inputs': {'noise': ['11', 0], 'guider': ['10', 0],
                          'sampler': ['9', 0], 'sigmas': ['8', 0],
                          'latent_image': ['7', 0]}},
        '13': {'class_type': 'VAEDecode',
               'inputs': {'samples': ['12', 0], 'vae': ['3', 0]}},
        '14': {'class_type': 'SaveImage',
               'inputs': {'filename_prefix': filename_prefix, 'images': ['13', 0]}},
    }


MAX_OUTPUT_MP = 1.5
_LATENT_MULTIPLE = 16


def fit_output_size(width, height, max_mp=MAX_OUTPUT_MP):
    """(w, h) keeping the source aspect ratio, scaled to at most `max_mp`
    megapixels and snapped to a multiple of 16. Never upscales a small source."""
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        w = h = 0
    if w <= 0 or h <= 0:
        return 1024, 1024
    budget = max(0.1, float(max_mp)) * 1_000_000
    if w * h > budget:
        scale = (budget / (w * h)) ** 0.5
        w, h = w * scale, h * scale
    snap = lambda v: max(_LATENT_MULTIPLE, int(round(v / _LATENT_MULTIPLE)) * _LATENT_MULTIPLE)
    return snap(w), snap(h)


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _denoise():
    """THE identity<->prompt dial: how much noise img2img adds over the reference.
    LOW = sticks to the reference (stronger likeness, less variety); HIGH = follows
    the prompt (looser likeness). Default 0.75 (a base img2img engine needs more
    denoise than an edit model before the prompt overrides the reference)."""
    return _clamp(cfg.get('zimage.denoise'), 0.1, 1.0, 0.75)


def _steps():
    return int(_clamp(cfg.get('zimage.steps'), 1, 50, 8.0))


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_zimage_edit(user_id, source_filename, edit_prompt, source_path=None,
                        extra_metadata=None, zimage_model=None):
    """Pre-size the reference, copy it into ComfyUI's input folder, build the
    img2img graph against what is ACTUALLY installed, and enqueue it. Returns the
    app job_id. Raises ZImageModelsMissing (via preflight) when an asset is absent,
    ValueError on a missing source, RuntimeError when ComfyUI isn't configured."""
    from PIL import Image
    if source_path is None:
        out_dir = cfg.comfyui_dir('output')
        if not out_dir:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(str(out_dir), source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f'source image not found: {source_filename}')

    preflight()
    unet = resolve_zimage_unet(zimage_model)
    clip = resolve_zimage_text_encoder()
    vae = resolve_zimage_vae()

    comfy_input_dir = _comfy_input_dir()
    uid = uuid.uuid4().hex[:8]
    comfy_input = f'zimage_source_{uid}.png'
    with Image.open(source_path) as im:
        im = im.convert('RGB')
        w, h = fit_output_size(*im.size)
        im.resize((w, h), Image.LANCZOS).save(os.path.join(comfy_input_dir, comfy_input))

    workflow = build_workflow(
        comfy_input, edit_prompt, unet=unet, clip=clip, vae=vae,
        seed=random.randint(0, 2 ** 64 - 1), steps=_steps(), denoise=_denoise(),
        # UNIQUE prefix per job: SaveImage numbers from what is in the output
        # folder and the app moves each result out right after completion, so a
        # shared prefix would re-issue the same name (the Klein tile-dup bug).
        filename_prefix=f'{user_id}_DatasetZImage_{uid}')

    job_id = str(uuid.uuid4())
    meta = {'model_name': 'zimage_turbo_dataset'}
    if extra_metadata:
        meta.update(extra_metadata)
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=edit_prompt,
                          job_id=job_id, metadata=meta)
    return job_id
