"""Z-Image — the THIRD local generation engine, next to Klein and Krea.

WHAT IT IS
----------
Z-Image (Tongyi-MAI, Apache-2.0) is a base text-to-image model, shipped as a fast
distilled TURBO build and a non-distilled BASE build. It has NO identity-edit
mechanism of its own (unlike Krea's edit LoRA + custom node pack) and there is no
Z-Image-Edit checkpoint in existence. Identity is therefore LOOSER than Klein or
Krea at any setting, and the UI states this honestly.

WHY THERE IS A GRAPH FAMILY AND NOT ONE GRAPH
---------------------------------------------
Plain img2img over the reference cannot vary a pose, and the reason is geometric.
The reference this engine gets is `ds.ref_filename` — the SQUARE HEAD CROP — and
the output canvas used to be derived from that same square. A head-shaped frame
initialised with a head returns a head at every denoise; 'full body, standing' came
back a bust. Composition and identity were riding the SAME channel (the init
latent), so the one denoise scalar could only trade one against the other.

Krea does not have that problem because it generates composition from an EMPTY
latent at denoise 1.0 and feeds identity through a SEPARATE channel
(Krea2EditGroundedEncode + Krea2EditModelPatch). We reproduce that split with stock
ComfyUI nodes by separating the two SPATIALLY instead of architecturally:

  close     face shots at a square aspect. img2img over the reference, at the
            SHOT's aspect (never upscaled past the source — see plan_shot).
  restage   bust/body and any non-square shot. The shot is built on a fresh canvas
            at its own aspect; the head crop is composited in; an elliptical,
            blurred keep-mask holds ONLY the head while the pose, outfit and
            background denoise fully. Composition free, identity anchored.
  backdrop  back shots with no full-frame original. No reference at all — a back
            view carries no face, and pasting one in produces a head-on-backwards
            artifact. Honest consequence: those shots carry no identity link.

THE GRAPH (core nodes only — no custom pack, unlike Krea)
--------------------------------------------------------
    UNETLoader ─┐                        BasicScheduler(denoise) ─┐
    CLIPLoader(type='lumina2') ─ CLIPTextEncode(±) ─ CFGGuider ─┐ │
    VAELoader ─ VAEEncode(reference) ── latent ─────────────────┼─┼─ SamplerCustomAdvanced ─ VAEDecode ─ SaveImage
    LoadImage(reference) ──────────────────────────────────────┘ │  ▲ KSamplerSelect(euler)  ▲ RandomNoise(seed)
    restage adds:  LoadImageMask ─ SetLatentNoiseMask ─────────────┘
    backdrop swaps LoadImage+VAEEncode for EmptySD3LatentImage.

TURBO vs BASE
-------------
Turbo is guidance-distilled: cfg is pinned to 1.0 and the negative branch is inert.
Base takes a real cfg, more steps and a real negative prompt, which is what buys
prompt adherence on restaged shots. Detection defaults to TURBO on purpose — see
zimage_variant.

Assets (3 required, no LoRA): UNET + text-encoder + VAE, plus the OPTIONAL Base
checkpoint. Because it is core-nodes-only, there is NO custom-node preflight (like
Klein, unlike Krea).
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
    setting, matched on BASENAME), then the build matching an explicitly chosen
    `zimage.variant`, then a 'turbo' build, then the first candidate.
    Deterministic — the same install always resolves the same file.

    The variant step matters: asking for Base while the resolver kept handing back
    the Turbo file would run a distilled model at cfg 4.0 / 28 steps, which is burnt
    output rather than a soft one. An explicit variant therefore steers the FILE too,
    not just the sampler settings."""
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
    want = str(cfg.get('zimage.variant') or 'auto').strip().lower()
    if want == 'base':
        for sub, names in folders:
            for n in names:
                low = n.lower()
                if 'turbo' not in low and 'distill' not in low:
                    return os.path.join(sub, n)
        logger.warning('zimage.variant=base but no non-turbo Z-Image build is on '
                       'disk — falling back to whatever is installed')
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


# The OPTIONAL Base checkpoint. Not in ZIMAGE_ASSETS/ZIMAGE_REQUIRED on purpose:
# it is a second checkpoint, not a missing dependency, and listing it there would
# turn every existing install's readiness probe red.
ZIMAGE_BASE_FILENAME = 'z_image_bf16.safetensors'


def zimage_base_installed():
    """Is a Z-Image BASE (non-distilled) build on disk? Drives the optional Setup
    row and the Base/Turbo copy — never gates the engine."""
    for _sub, names in _zimage_unet_folders():
        for n in names:
            low = n.lower()
            if 'turbo' in low or 'distill' in low:
                continue
            if low == ZIMAGE_BASE_FILENAME or '_base' in low or 'base_' in low:
                return True
    return False



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
                   steps=8, denoise=0.75, cfg_scale=1.0, negative_prompt='',
                   mask_image=None, empty_latent=None,
                   filename_prefix='zimage_edit'):
    """ComfyUI API-format graph in one of THREE shapes, built from
    ZImage_bigLove_ZT3_optimal.json (the advanced-sampler form Z-Image uses). Pure
    function of its arguments — no cfg read, no disk — so a test can assert the
    exact wiring without a ComfyUI. The reference/canvas is pre-sized by the caller
    (enqueue), so no scale node is needed here.

      * default        — img2img: LoadImage -> VAEEncode -> sampler  ('close')
      * `mask_image`   — img2img + LoadImageMask -> SetLatentNoiseMask: the masked
                         region is held while the rest is regenerated ('restage')
      * `empty_latent` — (w, h): text-to-image, no reference at all ('backdrop').
                         Mutually exclusive with `mask_image`.

    `cfg_scale` stays 1.0 for TURBO (guidance-distilled: it ignores anything else,
    and the negative branch is inert at cfg 1.0). Z-Image BASE is not distilled, so
    it takes a real cfg AND a real `negative_prompt`.

    RESTAGE — the mask convention is ComfyUI's, and it is the opposite of what the
    name suggests. `KSamplerX0Inpaint` computes `latent_mask = 1 - denoise_mask`
    and returns `out*denoise_mask + latent_image*latent_mask`, so mask **1 =
    regenerate**, mask **0 = preserve verbatim**, and values in between blend at
    every step. So the mask we ship is WHITE everywhere (regenerate the pose, the
    background, the outfit) except a soft ellipse over the head. It is read on the
    RED channel: LoadImageMask's `alpha` path inverts (`1 - alpha`) and a plain
    LoadImage MASK output would too — red does not."""
    steps = 8 if steps is None else max(1, int(steps))
    denoise = 0.75 if denoise is None else float(denoise)
    cfg_scale = 1.0 if cfg_scale is None else float(cfg_scale)
    if mask_image and empty_latent:
        raise ValueError('mask_image and empty_latent are mutually exclusive')
    graph = {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': unet, 'weight_dtype': 'default'},
              '_meta': {'title': 'Z-Image base model'}},
        '2': {'class_type': 'CLIPLoader',
              'inputs': {'clip_name': clip, 'type': 'lumina2'},
              '_meta': {'title': 'Qwen3-4B text encoder'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': vae}},
        '4': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': prompt, 'clip': ['2', 0]},
              '_meta': {'title': 'Positive'}},
        '5': {'class_type': 'CLIPTextEncode',
              'inputs': {'text': negative_prompt or '', 'clip': ['2', 0]},
              '_meta': {'title': 'Negative'}},
        '8': {'class_type': 'BasicScheduler',
              'inputs': {'scheduler': 'simple', 'steps': steps,
                         'denoise': denoise, 'model': ['1', 0]}},
        '9': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
        '10': {'class_type': 'CFGGuider',
               'inputs': {'cfg': cfg_scale, 'model': ['1', 0],
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
    if empty_latent:
        # No reference at all: nothing to LoadImage, nothing to encode.
        w, h = empty_latent
        graph['17'] = {'class_type': 'EmptySD3LatentImage',
                       'inputs': {'width': int(w), 'height': int(h),
                                  'batch_size': 1},
                       '_meta': {'title': 'Empty canvas (no reference)'}}
        graph['12']['inputs']['latent_image'] = ['17', 0]
        return graph

    graph['6'] = {'class_type': 'LoadImage', 'inputs': {'image': source_image}}
    graph['7'] = {'class_type': 'VAEEncode',
                  'inputs': {'pixels': ['6', 0], 'vae': ['3', 0]},
                  '_meta': {'title': 'Init latent (img2img)'}}
    if mask_image:
        graph['15'] = {'class_type': 'LoadImageMask',
                       'inputs': {'image': mask_image, 'channel': 'red'},
                       '_meta': {'title': 'Keep-mask (white = regenerate)'}}
        graph['16'] = {'class_type': 'SetLatentNoiseMask',
                       'inputs': {'samples': ['7', 0], 'mask': ['15', 0]},
                       '_meta': {'title': 'Anchor the head, free the rest'}}
        graph['12']['inputs']['latent_image'] = ['16', 0]
    return graph


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


def _base_steps():
    """Z-Image BASE is not distilled and needs ~28 steps. Deliberately NOT
    `zimage.steps`: every existing install has 8 saved there, and silently running
    Base at 8 steps would look like the engine is broken."""
    return int(_clamp(cfg.get('zimage.base_steps'), 1, 60, 28.0))


def _base_cfg():
    return _clamp(cfg.get('zimage.base_cfg'), 1.0, 10.0, 4.0)


def zimage_variant(unet_name=None) -> str:
    """'turbo' | 'base'. The `zimage.variant` setting wins; otherwise guess from the
    filename; TURBO is the fail-safe default. That asymmetry is deliberate — Base at
    cfg 1.0 / 8 steps is merely soft and obviously fixable, while Turbo at cfg 4.0 /
    28 steps is burnt garbage. Fail toward the recoverable error."""
    pick = str(cfg.get('zimage.variant') or 'auto').strip().lower()
    if pick in ('turbo', 'base'):
        return pick
    low = os.path.basename(str(unet_name or '')).lower()
    if 'turbo' in low or 'distill' in low:
        return 'turbo'
    if 'z_image_bf16' in low or '_base' in low or 'base_' in low:
        return 'base'
    return 'turbo'


# --- Shot geometry -----------------------------------------------------------
# Z-Image is a plain base model doing img2img, so — unlike Krea, whose edit LoRA was
# trained on same-size pairs and must keep the source's frame (krea_edit_helper:493)
# — it is free to render at the SHOT's aspect. That freedom is the whole point: the
# reference we get is `ref_filename`, the SQUARE head crop, and a square canvas can
# only ever come back a head no matter how high the denoise goes.

ASPECT_MAX_MP = 1.5

# How the graph family is picked. 'restage' composites the head onto a
# framing-correct canvas and holds only the head; 'backdrop' has no reference at all.
FRAMING_MODE = {'face': 'close', 'bust': 'restage',
                'body': 'restage', 'back': 'backdrop'}

# Pasted crop side, as a fraction of canvas HEIGHT. The crop is head+shoulders at
# REF_CROP_PAD=2.0, so a whole human figure is roughly 3.75x that side — which is
# what puts a 'body' figure inside the frame instead of cropped at the chest.
CROP_SIDE_FRAC = {'bust': 0.62, 'body': 0.26}
# Where the centre of the pasted square sits, as a fraction of canvas HEIGHT.
HEAD_CENTER_FRAC = {'bust': 0.42, 'body': 0.14}

# `close` sampler denoise = the zimage.denoise setting x this. Face shots want to
# stay near the reference; everything else restages at denoise 1.0 anyway.
DENOISE_BIAS = {'face': 0.80, 'bust': 1.0, 'body': 1.0, 'back': 1.0}


def aspect_size(aspect, max_mp=ASPECT_MAX_MP):
    """(w, h) for a 'w:h' string — the same strings face_variations.aspect_for_label
    emits — filling `max_mp` megapixels, both sides snapped to a multiple of 16.
    Anything unparseable falls back to square."""
    try:
        aw, ah = (float(x) for x in str(aspect).split(':', 1))
        if aw <= 0 or ah <= 0:
            raise ValueError
    except (TypeError, ValueError, AttributeError):
        aw = ah = 1.0
    budget = max(0.1, float(max_mp)) * 1_000_000
    scale = (budget / (aw * ah)) ** 0.5
    snap = lambda v: max(_LATENT_MULTIPLE, int(round(v / _LATENT_MULTIPLE)) * _LATENT_MULTIPLE)
    return snap(aw * scale), snap(ah * scale)


def denoise_for(framing, denoise):
    """Sampler denoise for a `close` shot: the zimage.denoise setting biased by
    framing. Takes the setting as an ARGUMENT rather than reading it, so plan_shot
    stays pure and the bias table is testable on its own."""
    return round(_clamp(float(denoise) * DENOISE_BIAS.get(framing, 1.0),
                        0.1, 1.0, 0.60), 3)


def keep_value(denoise):
    """How much of the head ellipse is REGENERATED on a restage shot, 0.0 = pasted
    verbatim. Driven by the same zimage.denoise dial so it keeps meaning the same
    thing on both paths: lower setting = tighter likeness."""
    return round(max(0.0, min(0.60, float(denoise) - 0.40)), 3)


def plan_shot(framing, aspect, src_size, *, denoise, body_source='crop',
              has_full_frame=False):
    """Everything the canvas renderer and the graph builder need for ONE shot.
    Pure — no disk, no cfg, no PIL — so the geometry is unit-testable on its own.

    Returns {'mode', 'canvas', 'paste', 'ellipse', 'keep', 'blur', 'denoise',
    'source'}; 'paste'/'ellipse' are None outside restage."""
    mode = FRAMING_MODE.get(framing, 'close')
    # A 'face' shot carrying a wide/tall override is a scene shot wearing a
    # close-up's framing tag — restage it rather than letterboxing a head.
    square = aspect_size(aspect) if aspect else None
    if mode == 'close' and square and square[0] != square[1] and framing in FRAMING_MODE:
        mode = 'restage'

    if mode == 'backdrop':
        # A back shot must NEVER get a frontal head pasted into it — that gives a
        # head-on-backwards artifact. Bias from the full frame when we kept one
        # (hair, build and palette all read from behind); otherwise generate free
        # and accept that the shot carries no identity link.
        w, h = aspect_size(aspect or '3:4')
        if has_full_frame:
            return {'mode': 'close', 'canvas': (w, h), 'paste': None,
                    'ellipse': None, 'keep': None, 'blur': 0,
                    'denoise': 0.90, 'source': 'original'}
        return {'mode': 'backdrop', 'canvas': (w, h), 'paste': None,
                'ellipse': None, 'keep': None, 'blur': 0,
                'denoise': 1.0, 'source': None}

    if mode == 'close':
        # Honour the shot's aspect, but keep fit_output_size's never-upscale rule:
        # a 512x512 reference must not be interpolated up into invented detail the
        # LoRA would then learn. So the aspect is respected, the AREA is not grown.
        if square:
            src_mp = max(1, src_size[0] * src_size[1]) / 1_000_000
            w, h = aspect_size(aspect, min(ASPECT_MAX_MP, src_mp))
        else:
            w, h = fit_output_size(*src_size)
        return {'mode': 'close', 'canvas': (w, h), 'paste': None, 'ellipse': None,
                'keep': None, 'blur': 0, 'denoise': denoise_for(framing, denoise),
                'source': 'crop'}

    w, h = square if square else aspect_size('3:4')
    side_frac = CROP_SIDE_FRAC.get(framing, CROP_SIDE_FRAC['body'])
    ctr_frac = HEAD_CENTER_FRAC.get(framing, HEAD_CENTER_FRAC['body'])
    side = int(max(64, min(round(side_frac * h), int(0.92 * min(w, h)))))
    px = (w - side) // 2
    py = int(round(ctr_frac * h)) - side // 2
    py = max(0, min(py, h - side))
    # face_crop_to_square_webp centres the head at ~0.45 of the square's height
    # (it shifts cy up by 10% of head height); the head itself is ~half the side.
    # The ellipse covers the HEAD ONLY — shoulders and the reference's background
    # stay at mask 1.0 and get regenerated, which is what kills the seam and stops
    # the reference's own backdrop leaking into every shot.
    cx = px + side // 2
    cy = py + int(round(0.45 * side))
    rx = max(8, int(round(0.30 * side)))
    ry = max(8, int(round(0.38 * side)))
    keep = keep_value(denoise)
    if framing == 'body':
        # A body head is ~18 latent px across — too little to carry a face unless
        # it is held tighter than a bust's.
        keep = round(keep * 0.6, 3)
    src = 'original' if (body_source == 'original' and has_full_frame) else 'crop'
    return {'mode': 'restage', 'canvas': (w, h), 'paste': (px, py, side),
            'ellipse': (cx, cy, rx, ry), 'keep': keep,
            'blur': max(12, int(round(0.10 * side))), 'denoise': 1.0,
            'source': src}


def fit_cover(im, size):
    """Resize `im` to exactly `size`, preserving its aspect by scaling to COVER and
    centre-cropping the overflow. A plain resize to a different aspect would squash
    the face — which matters now that the canvas follows the SHOT's aspect rather
    than the source's."""
    from PIL import Image
    tw, th = size
    sw, sh = im.size
    if sw <= 0 or sh <= 0:
        return im.resize((tw, th), Image.LANCZOS)
    scale = max(tw / sw, th / sh)
    rw, rh = max(tw, int(round(sw * scale))), max(th, int(round(sh * scale)))
    im = im.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - tw) // 2, (rh - th) // 2
    return im.crop((left, top, left + tw, top + th))


def render_canvas(src_path, plan, canvas_path, mask_path):
    """Write the composited init image AND its keep-mask for a restage shot.

    Canvas: flat 0.5 grey with the reference pasted at `plan['paste']`. Only the
    ellipse survives sampling, so the grey is scaffolding — it matches
    ImagePadForOutpaint's convention and gives a debuggable PNG.

    Mask: 8-bit RGB (NOT 'L', NOT RGBA — LoadImageMask reads the RED channel and
    the alpha path would invert), white = regenerate, the head ellipse filled with
    the keep value, then blurred. That blur IS the feather: core FeatherMask can
    only feather from a mask's outer edges, never an inset ellipse."""
    from PIL import Image, ImageDraw, ImageFilter
    w, h = plan['canvas']
    px, py, side = plan['paste']
    canvas = Image.new('RGB', (w, h), (128, 128, 128))
    with Image.open(src_path) as im:
        crop = im.convert('RGB')
        if plan.get('source') == 'original':
            # Full frame: scale so the FIGURE lands where the pasted square would,
            # then centre it on the same box.
            fw, fh = crop.size
            target_h = max(1, int(round(side * 3.75)))
            scale = target_h / max(1, fh)
            crop = crop.resize((max(1, int(round(fw * scale))), target_h), Image.LANCZOS)
            canvas.paste(crop, (px + side // 2 - crop.width // 2, py))
        else:
            canvas.paste(crop.resize((side, side), Image.LANCZOS), (px, py))
    canvas.save(canvas_path)

    cx, cy, rx, ry = plan['ellipse']
    mask = Image.new('RGB', (w, h), (255, 255, 255))
    v = int(round(max(0.0, min(1.0, plan['keep'])) * 255))
    ImageDraw.Draw(mask).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(v, v, v))
    mask.filter(ImageFilter.GaussianBlur(plan['blur'])).save(mask_path)



def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_zimage_edit(user_id, source_filename, edit_prompt, source_path=None,
                        extra_metadata=None, zimage_model=None, *,
                        framing=None, aspect=None, full_frame_path=None,
                        negative_prompt=None):
    """Plan the shot, write whatever init images it needs into ComfyUI's input
    folder, build the graph against what is ACTUALLY installed, and enqueue it.
    Returns the app job_id. Raises ZImageModelsMissing (via preflight) when an
    asset is absent, ValueError on a missing source, RuntimeError when ComfyUI
    isn't configured.

    The four keyword-only shot arguments all default to None, which plans a
    `close` shot at the source's own aspect — i.e. exactly the pre-graph-family
    behaviour, which is also what legacy DB rows with no `framing` get."""
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
    variant = zimage_variant(unet)

    comfy_input_dir = _comfy_input_dir()
    uid = uuid.uuid4().hex[:8]
    with Image.open(source_path) as im:
        src_size = im.size
    plan = plan_shot(framing, aspect, src_size, denoise=_denoise(),
                     body_source=str(cfg.get('zimage.body_source') or 'crop'),
                     has_full_frame=bool(full_frame_path))
    if plan['source'] == 'original' and full_frame_path:
        source_path = full_frame_path

    comfy_input = mask_input = empty_latent = None
    if plan['mode'] == 'backdrop':
        empty_latent = plan['canvas']
    elif plan['mode'] == 'restage':
        comfy_input = f'zimage_source_{uid}.png'
        mask_input = f'zimage_mask_{uid}.png'
        render_canvas(source_path, plan,
                      os.path.join(comfy_input_dir, comfy_input),
                      os.path.join(comfy_input_dir, mask_input))
    else:
        comfy_input = f'zimage_source_{uid}.png'
        with Image.open(source_path) as im:
            # COVER, not stretch: the canvas now follows the shot's aspect, so a
            # square head crop into a 3:4 frame must be cropped, never squashed.
            fit_cover(im.convert('RGB'), plan['canvas']).save(
                os.path.join(comfy_input_dir, comfy_input))

    workflow = build_workflow(
        comfy_input, edit_prompt, unet=unet, clip=clip, vae=vae,
        seed=random.randint(0, 2 ** 64 - 1),
        steps=_base_steps() if variant == 'base' else _steps(),
        denoise=plan['denoise'],
        # Turbo is guidance-distilled: cfg 1.0 and an inert negative branch. Base
        # is not, so it gets a real cfg and the caller's negative prompt.
        cfg_scale=_base_cfg() if variant == 'base' else 1.0,
        negative_prompt=(negative_prompt or '') if variant == 'base' else '',
        mask_image=mask_input, empty_latent=empty_latent,
        # UNIQUE prefix per job: SaveImage numbers from what is in the output
        # folder and the app moves each result out right after completion, so a
        # shared prefix would re-issue the same name (the Klein tile-dup bug).
        filename_prefix=f'{user_id}_DatasetZImage_{uid}')

    job_id = str(uuid.uuid4())
    meta = {'model_name': f'zimage_{variant}_dataset', 'zimage_mode': plan['mode']}
    if extra_metadata:
        meta.update(extra_metadata)
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=edit_prompt,
                          job_id=job_id, metadata=meta)
    return job_id
