"""Capability probes: what's actually reachable/configured right now.

Consumed by the Settings UI (engine cards, "Test connection" buttons) and by
feature gating elsewhere in the app. `_http_ok` is the single network seam —
every reachability probe goes through it so tests can patch one symbol.
`_import_ok` is the equivalent seam for the slow subprocess import-probes.
"""
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

from . import config as cfg
from .utils import comfy_fs

_CACHE_TTL = 30
_cache = None
_cache_ts = 0.0

_IMPORT_TTL = 600
# How long an UNKNOWN verdict (the probe never answered) is remembered. Short,
# because it must re-try soon against a warm import — but not zero: the Bank
# panel polls its readiness every ~2 s, and an uncached unknown meant a fresh
# 90 s `import torch` subprocess on EVERY poll.
_UNKNOWN_TTL = 60
# Budget for one cold import probe. Aligned with services.scoring_python
# .PROBE_TIMEOUT (90 s), which the repo already documents as the honest floor
# for a cold `import torch` behind an antivirus: at 60 s the SAME interpreter
# answered 'CUDA' to one probe and 'no answer' to the other.
_IMPORT_TIMEOUT = 90
_import_cache = {}  # key -> (ts, ok|None)  — None = unknown, kept briefly

_ZIMAGE_RE = re.compile(r'z[ -]?image', re.IGNORECASE)
# Aligned with klein_edit_helper / utils.comfyui (was missing '.sft', so the
# picker under-listed vs the resolvers). '.gguf' is listed but NOT loadable by the
# shipped graphs — they emit core `UNETLoader`, which has no '.gguf' support (core
# ComfyUI's supported_pt_extensions does not include it) and would need the
# ComfyUI-GGUF pack's separate `UnetLoaderGGUF` node, which nothing here emits.
# It stays listed so a .gguf a user already has is not silently invisible; what
# makes that honest is `utils.comfyui.unavailable_model_files`, which names the
# extension as the cause before anything is queued (naniii2352, Discord).
_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')


def _http_ok(url, timeout=3) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


def _import_ok(python: str, module_expr: str, timeout=_IMPORT_TIMEOUT):
    """True/False = the import deterministically succeeded/failed. None = TIMEOUT —
    unknown, NOT a proven absence. The very first `import rembg` after an install
    compiles numba/scikit-image caches while the antivirus scans 40 MB of fresh
    DLLs: measured ~20 s cold vs ~1 s warm — a 20 s timeout read as False showed
    'Person masks ✗' for 10 min right after a SUCCESSFUL install."""
    try:
        result = subprocess.run([python, '-c', module_expr], capture_output=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return False


def _cached_import_state(key: str, python: str, module_expr: str):
    """Three-valued, cached import probe: True / False / None.

    None means the probe DID NOT ANSWER — a cold-import timeout, not a proven
    absence. Callers that only gate a feature can flatten it to False
    (_cached_import); callers whose wrong answer costs something real — the
    GPU-exclusive window — must be able to tell the two apart.

    Both verdicts are cached, with very different lifetimes: a decided answer
    for _IMPORT_TTL (a venv does not change between two probes), an unknown for
    _UNKNOWN_TTL so it re-tries soon against a now-warm import WITHOUT spawning
    a fresh 90 s subprocess on every 2 s poll of the Bank panel."""
    now = time.time()
    cache_key = f'{key}:{python}:{module_expr}'
    cached = _import_cache.get(cache_key)
    if cached is not None:
        ttl = _IMPORT_TTL if cached[1] is not None else _UNKNOWN_TTL
        if now - cached[0] < ttl:
            return cached[1]
    ok = _import_ok(python, module_expr)
    _import_cache[cache_key] = (now, ok)
    return ok


def _cached_import(key: str, python: str, module_expr: str) -> bool:
    """The boolean gate: an unknown reads as not-ready (nothing is offered on a
    capability we could not prove), but it is no longer re-probed on every call."""
    return _cached_import_state(key, python, module_expr) is True


def probe_gemini() -> dict:
    ok = bool(cfg.secret('GEMINI_API_KEY'))
    return {'ok': ok, 'detail': 'key set' if ok else 'key missing'}


def probe_openai() -> dict:
    """ChatGPT engine readiness: a pay-per-use API key OR a connected ChatGPT
    subscription (Codex OAuth) both light the engine up."""
    from .services import chatgpt_oauth
    key = bool(cfg.secret('OPENAI_API_KEY'))
    sub = chatgpt_oauth.status()['connected']
    parts = (['key set'] if key else []) + (['subscription connected'] if sub else [])
    return {'ok': key or sub, 'detail': ' + '.join(parts) if parts else 'key missing'}


def probe_openrouter() -> dict:
    """OpenRouter engine readiness. Presence of the key only — same contract as
    probe_gemini. Deliberately NOT a live call: verifying the key would mean an
    authenticated request on every capability poll, and OpenRouter bills per
    request; a key that is set but out of credits is reported at generation time,
    with the provider's own words."""
    ok = bool(cfg.secret('OPENROUTER_API_KEY'))
    return {'ok': ok, 'detail': 'key set' if ok else 'key missing'}


def probe_comfyui() -> dict:
    api_url = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api_url:
        return {'ok': False, 'detail': 'comfyui.api_url not configured'}
    ok = _http_ok(f'{api_url}/history')
    return {'ok': ok, 'detail': api_url if ok else f'unreachable: {api_url}'}


def probe_ollama() -> dict:
    url = (cfg.get('ollama.url') or '').rstrip('/')
    if not url:
        return {'ok': False, 'detail': 'ollama.url not configured'}
    ok = _http_ok(f'{url}/api/tags')
    return {'ok': ok, 'detail': url if ok else f'unreachable: {url}'}


# Ollama install detection, EXECUTION-INDEPENDENT: it must answer "installed"
# even when the server is stopped. probe_ollama() above only sees a RUNNING
# server (HTTP probe), which made an installed-but-stopped Ollama read as absent
# — misleading. This pair lets the UI tell 'not installed' from 'installed but
# stopped' (→ offer a Start button) from 'running'.
_OLLAMA_WIN_BINARY = ('Programs', 'Ollama', 'ollama.exe')   # under %LOCALAPPDATA%


def _ollama_binary() -> str:
    """Absolute path to the Ollama CLI binary if installed, else ''. Two signals,
    neither of which needs the server running:
      1. ``shutil.which('ollama')`` — the official installer adds Ollama to PATH
         (Windows per-user, macOS/Linux /usr/local/bin), so this is the primary hit.
      2. Windows fallback: the per-user location the official installer writes to,
         ``%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe`` (verified against
         docs.ollama.com/windows) — covers a shell whose PATH was not refreshed
         since the install. First hit wins; never raises."""
    exe = shutil.which('ollama')
    if exe:
        return exe
    if os.name == 'nt':
        local = os.environ.get('LOCALAPPDATA')
        if local:
            cand = Path(local).joinpath(*_OLLAMA_WIN_BINARY)
            try:
                if cand.is_file():
                    return str(cand)
            except OSError:
                pass
    return ''


def probe_ollama_installed() -> dict:
    """Is the Ollama binary present on disk (independent of the server running)?
    The `installed` capability the UI reads alongside `reachable` to pick between
    an install guide (not installed) and a Start button (installed but stopped)."""
    path = _ollama_binary()
    return {'ok': bool(path), 'binary_path': path,
            'detail': path or 'ollama binary not found (PATH or default install location)'}


def _ollama_tags(url, timeout=3) -> list:
    """Model identifiers Ollama reports at /api/tags. Each entry is read from BOTH
    the `name` and `model` fields: across Ollama versions either one can be the field
    that carries the (namespaced) identifier, and builds exist that populate `model`
    while leaving `name` blank — reading a single field made a genuinely-pulled model
    read as absent (issue #7: a namespaced model present in the list, yet
    vision_model=no). Blanks are dropped and order is preserved. Network seam
    (patched in tests)."""
    try:
        resp = requests.get(f'{url}/api/tags', timeout=timeout)
        if resp.status_code >= 400:
            return []
        out, seen = [], set()
        for m in (resp.json().get('models') or []):
            if not isinstance(m, dict):
                continue
            for key in ('name', 'model'):
                v = (m.get(key) or '').strip()
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
        return out
    except Exception:
        return []


def _normalize_model_ref(ref: str) -> tuple:
    """Split an Ollama model reference into (repo, tag) for comparison, folding away
    only the two purely-transport differences that make the SAME model look different
    across `/api/tags` shapes:

      * a leading registry-host segment — `registry.ollama.ai/huihui_ai/x:t` or
        `localhost:5000/org/x:t` — is stripped (first path segment containing a '.'
        or a ':' is the host);
      * an implicit tag defaults to 'latest'.

    The publisher namespace and model name are PRESERVED ('huihui_ai/qwen3-vl-
    abliterated'), so an abliterated build never matches the vanilla one and two
    different publishers of the same name never collide. ('', '') for an empty ref.
    """
    ref = (ref or '').strip()
    if not ref:
        return ('', '')
    # The tag is the text after the LAST ':' — unless that ':' belongs to a host:port
    # (there's a '/' after it) or there is no ':' at all -> implicit 'latest'.
    repo, sep, tag = ref.rpartition(':')
    if not sep or '/' in tag:
        repo, tag = ref, 'latest'
    tag = tag or 'latest'
    segs = repo.split('/')
    if len(segs) > 1 and ('.' in segs[0] or ':' in segs[0]):   # leading registry host
        segs = segs[1:]
    return ('/'.join(segs), tag)


def _model_present(configured: str, names: list) -> bool:
    if not configured:
        return False
    if configured in names:                            # fast path: byte-exact match
        return True
    cfg_repo, cfg_tag = _normalize_model_ref(configured)
    if not cfg_repo:
        return False
    # A config value WITHOUT an explicit tag matches ANY tag of that repo (unchanged
    # semantics) — detected on the model segment so 'localhost:5000/x' isn't misread.
    cfg_has_tag = ':' in configured.rsplit('/', 1)[-1]
    for n in names:
        n_repo, n_tag = _normalize_model_ref(n)
        if n_repo and n_repo == cfg_repo and (not cfg_has_tag or n_tag == cfg_tag):
            return True
    return False


def probe_ollama_model(reachable=None) -> dict:
    # `reachable` lets probe() pass the reachability it already computed, so we
    # don't re-hit /api/tags a second time (and don't pay a second blocking
    # timeout when Ollama is configured-but-down). Called standalone -> we probe.
    url = (cfg.get('ollama.url') or '').rstrip('/')
    model = cfg.get('ollama.vision_model') or ''
    if not url:
        return {'ok': False, 'detail': 'ollama.url not configured'}
    if not model:
        return {'ok': False, 'detail': 'ollama.vision_model not configured'}
    if reachable is None:
        reachable = _http_ok(f'{url}/api/tags')        # gate on the stubbed seam first
    if not reachable:
        return {'ok': False, 'detail': f'ollama unreachable: {url}'}
    ok = _model_present(model, _ollama_tags(url))
    return {'ok': ok, 'detail': f'{model} ready' if ok else f'{model} not pulled'}


def probe_ollama_connection() -> dict:
    """The Settings 'Test' button for the Ollama card: an HONEST end-to-end check —
    the server is reachable AND the configured vision model is actually pulled.

    The old test target was probe_ollama (reachability only), so the green check
    disagreed with the Setup step / diagnostic model probe on the very same machine
    (issue #7: Test ✓ green while vision_model=no). This delegates to the SAME
    probe_ollama_model the Setup and diagnostic use, so all three now resolve through
    one seam and can never contradict — the same 'is_available() defers to the probe'
    unification as JoyCaption."""
    reach = probe_ollama()
    if not reach['ok']:
        return reach                              # not configured / unreachable — as-is
    return probe_ollama_model(reachable=True)


def ollama_diagnostic() -> dict:
    """Paste-safe Ollama snapshot for /api/diagnostic: the configured vision-model
    string alongside the model tags the probe actually sees at /api/tags. This is the
    pair a bug report needs to tell a genuine 'not pulled' from a name/shape mismatch
    (issue #7) — without it, a report can only say vision_model=no with no way to see
    that the model IS listed under a slightly different identifier. Model names are
    not secrets; the list is de-duplicated and capped (count + per-entry length) so a
    large local library can't bloat the pasted report."""
    url = (cfg.get('ollama.url') or '').rstrip('/')
    configured = cfg.get('ollama.vision_model') or ''
    tags = _ollama_tags(url) if url else []
    return {'vision_model': configured, 'tags_seen': [t[:80] for t in tags[:20]]}


def comfyui_runtime(timeout=3) -> dict:
    """Live ComfyUI runtime snapshot for /api/diagnostic: version, GPU + VRAM and
    the current queue depth. NETWORK — deliberately kept OUT of probe() (which must
    stay network-free, it runs on every /api/capabilities call); the diagnostic is a
    manual one-click action, so a couple of short GETs are fine here.

    Returns {} when ComfyUI isn't configured / doesn't answer. Paste-safe: only the
    ComfyUI version string, the GPU name + VRAM totals (GPU model is not identity)
    and the queue counts — never a path or a secret. Never raises."""
    api = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api:
        return {}
    out = {}
    try:
        r = requests.get(f'{api}/system_stats', timeout=timeout)
        if r.status_code == 200:
            j = r.json() or {}
            sysinfo = j.get('system') or {}
            out['version'] = ((sysinfo.get('comfyui_version') or '').strip() or None)
            devs = j.get('devices') or []
            if devs and isinstance(devs[0], dict):
                d0 = devs[0]
                name = (d0.get('name') or '').strip()
                if name:
                    out['gpu'] = name[:60]
                total = d0.get('vram_total')
                free = d0.get('vram_free')
                if isinstance(total, (int, float)) and total > 0:
                    out['vram_total_gb'] = round(total / 1024 ** 3, 1)
                if isinstance(free, (int, float)) and free >= 0:
                    out['vram_free_gb'] = round(free / 1024 ** 3, 1)
    except Exception:
        pass
    try:
        r = requests.get(f'{api}/queue', timeout=timeout)
        if r.status_code == 200:
            j = r.json() or {}
            out['queue_running'] = len(j.get('queue_running') or [])
            out['queue_pending'] = len(j.get('queue_pending') or [])
    except Exception:
        pass
    return out


def clear_import_cache() -> None:
    """Drop cached import-probe results and the main probe cache so the next
    probe re-checks freshly installed packages instead of a stale 600s 'False'."""
    global _cache, _cache_ts
    _import_cache.clear()
    _cache = None
    _cache_ts = 0.0


# Where an ai-toolkit checkout keeps the Python that runs it. There is NO single
# answer: the README's `venv/`, a `.venv/`, a conda env, or a portable bundle that
# ships `python_embeded/python.exe` (a community easy-install script does exactly
# that — reported on Reddit by Psyko_2000). So we do not guess a layout: we knock
# on the folder and on each of its immediate sub-folders with every interpreter
# shape the app already knows (`scoring_python._INTERPRETER_SPOTS`) and keep the
# ones that answer as a real file. An empty list means "we found nothing here",
# which is a fact the UI can state honestly — never a claim that a venv is the
# only way. Nothing is executed and nothing is written; it is a stat() sweep one
# level deep, computed only for a checkout whose interpreter we could not resolve.
def aitoolkit_python_candidates(root, limit: int = 4) -> list:
    from .services.scoring_python import interpreters_in
    try:
        root = Path(root)
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    out, seen = [], set()
    for folder in (root, *children[:64]):
        for cand in interpreters_in(folder):
            key = os.path.normcase(str(cand))
            if key in seen:
                continue
            seen.add(key)
            out.append(str(cand))
            if len(out) >= limit:
                return out
    return out


def probe_aitoolkit() -> dict:
    d = cfg.aitoolkit_path('dir')
    if not d:
        return {'ok': False, 'detail': 'aitoolkit.dir not configured',
                'has_run': False, 'python_candidates': []}
    venv_python = cfg.aitoolkit_path('venv_python')
    has_run = (d / 'run.py').exists()
    # is_file(), NOT exists(): the training launch gate (lora_training.is_installed)
    # checks is_file(), so a venv_python that resolves to a directory or a broken
    # link would make the diagnostic report ai-toolkit=yes while training still
    # says "not installed". Keep the two in lockstep so the diagnostic never lies.
    ok = has_run and bool(venv_python) and venv_python.is_file()
    if ok:
        return {'ok': True, 'detail': str(d), 'has_run': True,
                'python_candidates': []}
    if has_run:
        # The folder IS an ai-toolkit checkout — we just could not see which
        # Python runs it (conda/uv/system/portable installs all land here).
        # State the finding and hand over both ways out.
        found = aitoolkit_python_candidates(d)
        detail = (f'ai-toolkit found at {d} but no Python interpreter found '
                  'inside — create a venv there, or set its Python interpreter '
                  'in Settings → Local tools')
        if found:
            detail += f' (candidate: {found[0]})'
        return {'ok': False, 'detail': detail, 'has_run': True,
                'python_candidates': found}
    return {'ok': False, 'detail': f'invalid aitoolkit dir: {d}',
            'has_run': False, 'python_candidates': []}


# JoyCaption's runtime deps that ai-toolkit does NOT ship: the training venv has
# torch/torchvision, but joycaption_infer.py also needs transformers (AutoTokenizer
# / LlavaForConditionalGeneration), bitsandbytes (the NF4 4-bit load) and accelerate
# (required by from_pretrained with a quantization_config). Missing any of these is
# the ModuleNotFoundError users hit (issue #6). One import expr = one cached probe.
_JOYCAPTION_IMPORTS = 'import transformers, bitsandbytes, accelerate'
_JOYCAPTION_INSTALL = 'transformers bitsandbytes accelerate'


def probe_joycaption(aitoolkit: dict | None = None) -> dict:
    """Honest JoyCaption readiness. The old probe declared it available on the mere
    existence of the script + a configured ai-toolkit, so the app offered JoyCaption
    and then crashed with `ModuleNotFoundError: No module named 'transformers'` when
    the batch actually ran (issue #6). This checks the ai-toolkit venv can really
    import the captioning deps, through the cached subprocess seam so probe() stays
    fast and network-free (a per-probe subprocess would be unacceptable).

    `detail` names what to do: the exact `<venv_python> -m pip install …` command so
    the user can fix it without reading a stack trace. NEVER installs anything."""
    if aitoolkit is None:
        aitoolkit = probe_aitoolkit()
    if not aitoolkit['ok']:
        return {'ok': False, 'detail': aitoolkit['detail']}
    script = cfg.BACKEND_DIR / 'infer' / 'joycaption_infer.py'
    if not script.exists():
        return {'ok': False, 'detail': f'{script.name} not found'}
    venv_python = cfg.aitoolkit_path('venv_python')
    ok = _cached_import('joycaption', str(venv_python), _JOYCAPTION_IMPORTS)
    if ok:
        return {'ok': True, 'detail': 'JoyCaption deps import OK'}
    return {'ok': False,
            'detail': (f'JoyCaption deps ({_JOYCAPTION_INSTALL.replace(" ", ", ")}) '
                       f'are not importable in the ai-toolkit venv — run: '
                       f'"{venv_python}" -m pip install {_JOYCAPTION_INSTALL}')}


VAST_API_BASE = 'https://console.vast.ai/api/v0'


def probe_vast() -> dict:
    """Live check of the vast.ai API key (used by the Settings 'Test' button).
    The capability gate itself is key-presence only — probe() must stay
    network-free for this entry (it runs on every /api/capabilities call)."""
    key = cfg.secret('VAST_API_KEY')
    if not key:
        return {'ok': False, 'detail': 'API key missing'}
    try:
        r = requests.get(f'{VAST_API_BASE}/users/current/',
                         headers={'Authorization': f'Bearer {key}'}, timeout=8)
        if r.status_code == 200:
            email = (r.json() or {}).get('email') or 'account'
            return {'ok': True, 'detail': f'connected as {email}'}
        return {'ok': False, 'detail': f'vast.ai returned HTTP {r.status_code}'}
    except Exception as e:
        return {'ok': False, 'detail': f'unreachable: {e}'}


def probe_face_scoring() -> dict:
    python = cfg.get('face_scoring.python') or sys.executable
    ok = _cached_import('face_scoring', python, 'import insightface, onnxruntime')
    return {'ok': ok, 'detail': 'insightface + onnxruntime import OK' if ok else 'import failed'}


def face_gpu_available() -> bool:
    """True only when the face interpreter can run InsightFace on CUDA — i.e.
    onnxruntime exposes CUDAExecutionProvider (needs onnxruntime-gpu + a working
    CUDA/cuDNN runtime). The stock face_scoring extra ships CPU onnxruntime, so
    this is False until the user installs onnxruntime-gpu into that interpreter.
    Same cached subprocess probe as the import checks (exit 0 == available)."""
    python = cfg.get('face_scoring.python') or sys.executable
    return _cached_import(
        'face_gpu', python,
        "import onnxruntime,sys; "
        "sys.exit(0 if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else 1)")


def bank_scoring_gpu_available() -> bool:
    """True only when the bank-scoring interpreter can actually run torch on
    CUDA. The scoring child picks its own device (``cuda if
    torch.cuda.is_available()``), and the PARENT has to know the same answer:
    it decides whether to take the GPU-exclusive window, which unloads ComfyUI
    and blocks a training start for the whole pass. The stock extra installs
    CPU-only torch, so this is False until the user puts a CUDA build in that
    interpreter — and a pass that never touches the GPU must never hold it.

    UNKNOWN is NOT False here. 'the probe did not answer' and 'torch has no
    CUDA' used to collapse into the same answer, and they have opposite costs:
    the child decides on its own with `torch.cuda.is_available()`, so on a
    machine that HAS a card an unanswered probe means we may be leaving the GPU
    unprotected while the pass takes it — ComfyUI still loaded, a training start
    still allowed. So an unknown resolves to 'is there a card at all' (cached
    nvidia-smi read): a card-less machine still never holds a window it cannot
    use, and a machine with one is protected until the probe answers."""
    python = cfg.get('bank_scoring.python') or sys.executable
    state = _cached_import_state(
        'bank_scoring_gpu', python,
        'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)')
    if state is None:
        return gpu_vram_gb() is not None
    return state


def probe_masks() -> dict:
    python = cfg.get('masks.python') or sys.executable
    ok = _cached_import('masks', python, 'import rembg')
    return {'ok': ok, 'detail': 'rembg import OK' if ok else 'import failed'}


def probe_bank_scoring() -> dict:
    """Bank scoring extra (CLIP aesthetic + NSFW + style). Dedicated interpreter
    key (bank_scoring.python), else the app's own. Same subprocess-import probe as
    the other ML extras — torch/open_clip/transformers must all import. When False,
    the bank's Score button is disabled with an install hint (never a mute ✗)."""
    python = cfg.get('bank_scoring.python') or sys.executable
    ok = _cached_import('bank_scoring', python, 'import torch, open_clip, transformers')
    return {'ok': ok,
            'detail': 'torch + open_clip + transformers import OK' if ok else 'import failed'}


def probe_watermark_inpaint() -> dict:
    """LaMa inpainting availability (simple-lama-inpainting, ML extra). Dedicated
    interpreter key, else reuse the ML python (masks.python) then sys.executable —
    same subprocess-probe pattern/timeout handling as probe_masks. When False the
    Clean pass still runs crop-only (LaMa-routed images are skipped, not failed)."""
    python = cfg.get('watermark.python') or cfg.get('masks.python') or sys.executable
    ok = _cached_import('watermark', python, 'import simple_lama_inpainting')
    return {'ok': ok, 'detail': 'simple-lama-inpainting import OK' if ok else 'import failed'}


# Prebuilt wheels for the ML extras (insightface 0.7.3, numpy<2, onnxruntime,
# rembg, opencv) exist for CPython 3.10–3.12 only. On a newer interpreter (3.13+)
# there is no numpy<2 / insightface wheel, so `pip install -r requirements-ml.txt`
# falls back to source builds that can't resolve (numpy build-dep clash) — the
# cryptic failure a fresh-clone user hits. Surface the version so the setup can
# warn UP FRONT instead of after a 200-line pip traceback.
_ML_PY_MIN = (3, 10)
_ML_PY_MAX = (3, 12)


def python_ml_status() -> dict:
    """Version of THIS interpreter (the one `ml_extras` installs into via
    sys.executable) and whether it is inside the wheel-supported ML range."""
    v = sys.version_info
    return {
        'version': f'{v.major}.{v.minor}.{v.micro}',
        'ml_supported': _ML_PY_MIN <= (v.major, v.minor) <= _ML_PY_MAX,
        'ml_range': f'{_ML_PY_MIN[0]}.{_ML_PY_MIN[1]}–{_ML_PY_MAX[0]}.{_ML_PY_MAX[1]}',
    }


def probe_scrape_deps() -> dict:
    """The scraper's optional Python deps (requirements-scrape.txt). find_spec
    only (no import cost): the scrape stack runs IN-PROCESS, so the app's own
    interpreter is the one that must see the packages. curl_cffi + gallery_dl
    are the two hard requirements (picazor/civitai fetch, gallery enumeration);
    bs4/cloudscraper/instaloader ride along in the same install. Every module the
    scrape stack imports belongs here: an omission reads as "installed" while the
    source that needs it still raises at runtime (instaloader did, until 2026-07)."""
    import importlib.util
    missing = [m for m in ('curl_cffi', 'gallery_dl', 'bs4', 'cloudscraper', 'instaloader')
               if importlib.util.find_spec(m) is None]
    return {'ok': not missing,
            'detail': 'scrape deps OK' if not missing else f"missing: {', '.join(missing)}"}


def _model_files(folder) -> list:
    try:
        if not folder.is_dir():
            return []
        return sorted(
            p.name for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _MODEL_SUFFIXES
        )
    except OSError:
        return []


def _scan_models() -> dict:
    # Roots come from the SAME resolver ComfyUI uses (base <models> folders + any
    # extra_model_paths.yaml roots), so the picker/probe list exactly what a running
    # ComfyUI would load. With no yaml the roots are the historical [unet,
    # diffusion_models] / [checkpoints], so the output is byte-for-byte unchanged.
    from .services import comfy_model_paths
    result = {'zimage': [], 'sdxl': [], 'krea': [], 'klein': []}
    try:
        models_dir = cfg.comfyui_dir('models')
    except Exception:
        models_dir = None
    # krea is historically scanned ONLY from the base <models>/unet folder; track it
    # so an extra root (treated like diffusion_models) doesn't change that bucket.
    unet_default = os.path.normpath(str(models_dir / 'unet')) if models_dir else None

    for root in comfy_model_paths.search_roots('diffusion_models'):
        root_path = Path(root)
        try:
            subfolders = [p for p in root_path.iterdir() if p.is_dir()]
        except OSError:
            continue
        krea_eligible = (os.path.normpath(root) == unet_default)
        for sub in subfolders:
            name = sub.name
            if _ZIMAGE_RE.search(name):
                result['zimage'].extend(_model_files(sub))
            # Any 'klein'-named subfolder counts: shared installs keep e.g.
            # diffusion_models/'Flux2 klein'/ (the KV variant) next to our canonical
            # unet/klein/ download — hiding it made the picker blind to models the
            # user already owns.
            elif 'klein' in name.lower():
                result['klein'].extend(_model_files(sub))
            elif krea_eligible and name.lower().startswith('krea'):
                result['krea'].extend(_model_files(sub))
        # Flat / Stability-Matrix layouts drop the model straight INTO
        # diffusion_models/ with no klein/ subfolder — scan the root's own files
        # too and bucket the 'klein'-named ones. These are bare names (no prefix),
        # which is exactly what UNETLoader loads for a file at the root of a
        # registered folder. Mirrors klein_edit_helper._klein_unet_folders so the
        # picker lists only what the resolver can build.
        for name in _model_files(root_path):
            if 'klein' in name.lower():
                result['klein'].append(name)

    result['klein'] = sorted(set(result['klein']))
    sdxl = []
    for root in comfy_model_paths.search_roots('checkpoints'):
        sdxl.extend(_model_files(Path(root)))
    result['sdxl'] = sdxl
    return result


# --- Auto-detection (Setup wizard) -----------------------------------------
# Discover already-installed tools so the wizard can fill config itself. Two
# signals: a REACHABLE default port (safe to auto-apply — it answered) and a
# folder found on disk (a guess → the UI confirms before writing it).
_OLLAMA_DEFAULT_URL = 'http://127.0.0.1:11434'
_COMFYUI_DEFAULT_URL = 'http://127.0.0.1:8188'


def _common_roots() -> list:
    home = Path.home()
    candidates = [Path('C:/'), Path('D:/'), home, home / 'Downloads', home / 'Desktop',
                  home / 'projects', home / 'source' / 'repos', Path('C:/tools')]
    out, seen = [], set()
    for r in candidates:
        try:
            if r not in seen and r.is_dir():
                seen.add(r)
                out.append(r)
        except OSError:
            continue
    return out


def _find_install_dir(names, marker) -> str:
    """Shallow scan of common roots for a folder named in `names` satisfying
    `marker(path)`. First hit as a string, else ''. Shallow (root/name only) to
    stay fast — a deep recursive walk of C:\\ would be far too slow for a probe."""
    for root in _common_roots():
        for name in names:
            cand = root / name
            try:
                if cand.is_dir() and marker(cand):
                    return str(cand)
            except OSError:
                continue
    return ''


def _detect_ollama() -> dict:
    if not _http_ok(f'{_OLLAMA_DEFAULT_URL}/api/tags'):
        return {}
    out = {'url': _OLLAMA_DEFAULT_URL}
    names = _ollama_tags(_OLLAMA_DEFAULT_URL)
    vls = [n for n in names if 'vl' in (n or '').lower() or 'vision' in (n or '').lower()]
    # Preference among installed vision models. The uncensored *abliterated* build wins
    # first: the app's describe/caption work is NSFW-heavy and the vanilla qwen3-vl
    # refuses it outright, so an abliterated model must beat a censored one even when the
    # censored one is an -instruct tag. WITHIN a tier we still prefer -instruct over the
    # Thinking variant (Thinking reasons out loud instead of captioning; see
    # get_vision_model), then anything non-thinking. First match wins (order preserved).
    lo = [(n, (n or '').lower()) for n in vls]
    vl = (next((n for n, low in lo if 'abliterated' in low and 'instruct' in low), '')
          or next((n for n, low in lo if 'abliterated' in low), '')
          or next((n for n, low in lo if 'instruct' in low), '')
          or next((n for n, low in lo if 'thinking' not in low), '')
          or (vls[0] if vls else ''))
    if vl:
        out['vision_model'] = vl
    return out


# --- GPU VRAM probe (nvidia-smi, cached) ---------------------------------------
_gpu_cache = {'ts': 0.0, 'gb': None}
_GPU_TTL = 600


def gpu_vram_gb():
    """Total VRAM of GPU 0 in GB via nvidia-smi, cached 10 min. None when it can't
    be determined (no NVIDIA GPU / nvidia-smi absent) — callers must treat None
    as 'unknown', never as 0 (an unknown GPU must not trigger OOM warnings)."""
    import subprocess
    now = time.time()
    if _gpu_cache['ts'] and (now - _gpu_cache['ts']) < _GPU_TTL:
        return _gpu_cache['gb']
    gb = None
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if proc.returncode == 0:
            first = (proc.stdout or '').strip().splitlines()
            if first:
                gb = round(float(first[0].strip()) / 1024, 1)   # MiB -> GB
    except (OSError, ValueError, subprocess.TimeoutExpired):
        gb = None
    _gpu_cache.update(ts=now, gb=gb)
    return gb


# --- ai-toolkit torch probe (what actually trains) -----------------------------
# ai-toolkit runs in ITS OWN venv, which LDS never installs — it only reads the
# interpreter the user pointed at. Whether that venv's torch carries kernels for
# the local GPU is invisible from here, and getting it wrong is silent: RTX 50
# (Blackwell, sm_120) + a stable wheel = `is_available()` True, then a hard
# "no kernel image is available for execution on the device" at the first real
# computation. So: probe the venv, but only when it can matter.
#
# COST DISCIPLINE. `import torch` in a cold venv costs seconds, so the expensive
# probe is gated behind a ~100 ms nvidia-smi capability read: a GPU below
# compute 10.0 (everything up to Ada / RTX 40) can never hit the trap and pays
# nothing at all. What we do run is cached 10 min — a venv does not change
# between two runs.
_cc_cache = {'ts': 0.0, 'cc': None}
_TORCH_PROBE_TTL = 600
_torch_probe_cache = {}   # interpreter path -> (ts, info)

# First capability major that stable wheels may not cover (Blackwell = 12).
# 10 is deliberately lower than 12: it keeps the gate honest if a future
# generation lands before the wheels do.
_RISKY_CC_MAJOR = 10

_TORCH_PROBE_CODE = (
    'import json, torch\n'
    'cap = name = None\n'
    'try:\n'
    '    if torch.cuda.is_available():\n'
    '        cap = list(torch.cuda.get_device_capability(0))\n'
    '        name = torch.cuda.get_device_name(0)\n'
    'except Exception:\n'
    '    pass\n'
    'print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,\n'
    '                  "capability": cap, "device_name": name,\n'
    '                  "arch_list": list(torch.cuda.get_arch_list())}))\n'
)


def gpu_compute_capability():
    """(major, minor) of GPU 0 via nvidia-smi, cached 10 min. None = unknown."""
    now = time.time()
    if _cc_cache['ts'] and (now - _cc_cache['ts']) < _GPU_TTL:
        return _cc_cache['cc']
    cc = None
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if proc.returncode == 0:
            lines = (proc.stdout or '').strip().splitlines()
            if lines:
                major, _, minor = lines[0].strip().partition('.')
                cc = (int(major), int(minor or 0))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        cc = None
    _cc_cache.update(ts=now, cc=cc)
    return cc


def _torch_probe(python: str, timeout=90):
    """Raw torch facts from `python`, as a dict, or None. None is UNKNOWN — torch
    not importable, interpreter broken, cold-import timeout — never a claim."""
    try:
        proc = subprocess.run([python, '-c', _TORCH_PROBE_CODE],
                              capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        info = json.loads(((proc.stdout or '').strip().splitlines() or [''])[-1])
    except Exception:
        return None
    return info if isinstance(info, dict) else None


def aitoolkit_torch_info():
    """torch/GPU facts from the ai-toolkit venv — the interpreter that TRAINS —
    or None when we cannot know (no ai-toolkit, no NVIDIA GPU, GPU old enough to
    be covered by every wheel, torch not importable, probe timeout). Callers must
    treat None as 'no information', never as a verdict."""
    cc = gpu_compute_capability()
    if cc is None or cc[0] < _RISKY_CC_MAJOR:
        return None                       # cheap exit: no torch import at all
    python = cfg.aitoolkit_path('venv_python')
    if not python or not Path(python).is_file():
        return None
    key = str(python)
    now = time.time()
    hit = _torch_probe_cache.get(key)
    if hit and (now - hit[0]) < _TORCH_PROBE_TTL:
        return hit[1]
    info = _torch_probe(key)
    if info is None:
        return None      # a cold-import timeout must not be cached as a fact
    _torch_probe_cache[key] = (now, info)
    return info


def _is_comfyui_dir(d) -> bool:
    """A real ComfyUI install: classic (main.py at the root) OR the Desktop
    app's basedir (models/ + custom_nodes/, no main.py — a user had to
    symlink main.py to pass the old check). Everything the app does with this
    folder is SCAN models/, so that is the hard requirement."""
    try:
        if not (d / 'models').is_dir():
            return False
        return (d / 'main.py').exists() or (d / 'custom_nodes').is_dir()
    except OSError:
        return False


def resolve_comfyui_base(path: str) -> dict:
    """Resolve a user-entered ComfyUI folder to the one that actually holds main.py +
    models/ (which is what every base/model lister scans). Handles the common
    portable-bundle mistake: users point at ...\\ComfyUI_windows_portable, but main.py
    and models/ live one level down in ...\\ComfyUI_windows_portable\\ComfyUI. Without
    this, comfyui.base_dir\\models never exists -> "No SDXL checkpoint found" even though
    ComfyUI is running.

    Returns {valid, resolved, nested}: `nested` is True when we descended into a child
    ComfyUI/ (the caller can then auto-correct base_dir to `resolved`)."""
    if not path:
        return {'valid': False, 'resolved': '', 'nested': False}
    p = Path(path)
    if _is_comfyui_dir(p):
        return {'valid': True, 'resolved': str(p), 'nested': False}
    child = p / 'ComfyUI'
    if _is_comfyui_dir(child):
        return {'valid': True, 'resolved': str(child), 'nested': True}
    return {'valid': False, 'resolved': str(p), 'nested': False}


def classify_comfyui_dir(path: str) -> dict:
    """Rich verdict on a user-entered ComfyUI folder, so the Setup wizard can say
    something ACTIONABLE the moment the field is edited (before any save) instead of
    a blanket "invalid". `resolve_comfyui_base` only splits valid/nested/other; this
    keeps that split and additionally names WHY a folder isn't ComfyUI:

      status ∈
        'empty'       — nothing typed yet (the caller drives the skip flow).
        'valid'       — the folder itself is a ComfyUI install.
        'nested'      — the folder isn't, but ``<folder>/ComfyUI`` is (the launcher/
                        portable-wrapper mistake). `suggestion` = that child, to adopt.
        'missing'     — the path doesn't exist on disk.
        'empty_dir'   — the directory exists but is empty.
        'not_comfyui' — the directory (or a file at that path) exists and has content,
                        but holds no main.py/models/ and no child ComfyUI.

    `resolved` is the path a valid/nested verdict would adopt (child for nested,
    the folder itself otherwise). Never raises — a filesystem hiccup degrades
    to 'not_comfyui' rather than throwing into the request.

    Every verdict also carries `input_check` = {path, ok, problem}: "this IS a
    ComfyUI install" was only ever half the question, and the wizard used to
    certify the half it could see. The other half is whether the app can actually
    HAND FILES to that install — every local engine copies its source into
    `input/`. When ComfyUI runs in another container that folder is not shared by
    default, so the wizard went green and the first generation died on a bare 500
    (reported on Discord by nofaceman). `ok` is None when there is nothing to probe
    (no valid folder yet); a False NEVER blocks the wizard — someone may well
    configure the app before mounting their volumes."""
    raw = (path or '').strip()
    if not raw:
        return {'status': 'empty', 'resolved': '', 'suggestion': '',
                'input_check': _input_check('')}
    p = Path(raw)
    if _is_comfyui_dir(p):
        return {'status': 'valid', 'resolved': str(p), 'suggestion': '',
                'input_check': _input_check(str(p))}
    child = p / 'ComfyUI'
    if _is_comfyui_dir(child):
        return {'status': 'nested', 'resolved': str(child), 'suggestion': str(child),
                'input_check': _input_check(str(child))}
    try:
        exists, is_dir = p.exists(), p.is_dir()
    except OSError:
        exists, is_dir = False, False
    if not exists:
        return {'status': 'missing', 'resolved': str(p), 'suggestion': '',
                'input_check': _input_check('')}
    if is_dir:
        try:
            is_empty = not any(p.iterdir())
        except OSError:
            is_empty = False
        if is_empty:
            return {'status': 'empty_dir', 'resolved': str(p), 'suggestion': '',
                    'input_check': _input_check('')}
    # A file at that path, or a non-empty folder that simply isn't a ComfyUI checkout.
    return {'status': 'not_comfyui', 'resolved': str(p), 'suggestion': '',
            'input_check': _input_check('')}


def _input_check(base_dir: str) -> dict:
    """Can this process actually put a file in the input/ folder of the ComfyUI at
    `base_dir`? Honours a SAVED `comfyui.input_dir` override, so the wizard judges
    the folder the app would really use rather than a layout assumption.
    {'path','ok','problem'}; ok=None = nothing probed. Never raises."""
    if not base_dir:
        return {'path': '', 'ok': None, 'problem': ''}
    try:
        override = cfg.get('comfyui.input_dir') or ''
    except Exception:
        override = ''
    try:
        target = cfg.resolve_comfyui_dir('input', base_dir, override)
        path = str(target) if target else ''
        verdict = comfy_fs.probe_folder('input', path)
    except Exception:
        return {'path': '', 'ok': None, 'problem': ''}
    return {'path': comfy_fs.safe_path(path), 'ok': verdict['ok'],
            'problem': verdict['problem']}


def classify_comfyui_folders(base_dir: str, overrides: dict | None = None) -> dict:
    """Resolve the four ComfyUI working folders for a candidate (possibly unsaved)
    ComfyUI section, and say for each one where the path came from and whether it is
    actually there. Feeds the Settings preview so an override is never a leap of
    faith: an empty field SHOWS the derived path it falls back to, and a typed path
    that does not exist says so instead of failing silently at generation time.

    Returns {<config key>: {kind, source, resolved, exists, usable, problem}} where
      source ∈ 'override' (the field is filled) | 'derived' (from the install dir)
               | 'unset'  (no install dir and no override — nothing to resolve).

    `exists` alone certifies half the contract: a folder can be there and still be
    unusable from THIS process — the case that costs the most (ComfyUI in another
    container, an input/ mounted read-only) because the URL test goes green and the
    first generation dies on a copy. `usable` is the other half: for the folders the
    app WRITES into it is a real write-then-delete probe, for the others a read
    check. True/False, or None when there is nothing to probe (no path, or the path
    is not on disk — `exists` already says that). `problem` is the sentence to show.
    Read-only apart from the probe file it removes again, never raises."""
    overrides = overrides or {}
    out = {}
    for kind in cfg.COMFY_DIR_KINDS:
        key, _sub = cfg._COMFY_DERIVED[kind]
        explicit = str(overrides.get(key) or '').strip()
        p = cfg.resolve_comfyui_dir(kind, base_dir, explicit)
        resolved = str(p) if p else ''
        try:
            exists = bool(resolved) and Path(resolved).is_dir()
        except OSError:
            exists = False
        usable, problem = None, ''
        if exists:
            verdict = comfy_fs.probe_folder(kind, resolved)
            usable, problem = verdict['ok'], verdict['problem']
        out[key] = {'kind': kind,
                    'source': 'override' if explicit else ('derived' if resolved else 'unset'),
                    'resolved': resolved, 'exists': exists,
                    'usable': usable, 'problem': problem}
    return out


# ComfyUI takes its custom folders on the COMMAND LINE only (--input-directory,
# --output-directory, --models-directory); there is no config file to read. It does,
# however, echo its own argv back in /system_stats.system.argv, so the running
# instance can be ASKED what it was started with instead of guessing a layout.
_COMFY_ARGV_FLAGS = {'--output-directory': 'output_dir', '--input-directory': 'input_dir',
                     '--models-directory': 'models_dir'}


def parse_comfy_argv_dirs(argv) -> dict:
    """Extract the folder overrides ComfyUI was launched with from its own argv.

    Both argparse spellings are accepted (`--input-directory X` and
    `--input-directory=X`). RELATIVE paths are deliberately DROPPED: they resolve
    against ComfyUI's working directory, which we do not know, and this app never
    guesses a path by convention. `--base-directory` is likewise not turned into
    input/output suggestions — the install-directory field already derives those, and
    inventing them here would be a layout assumption, not an answer. Never raises."""
    out = {}
    if not isinstance(argv, (list, tuple)):
        return out
    items = [str(a) for a in argv]
    for i, tok in enumerate(items):
        flag, _, inline = tok.partition('=')
        key = _COMFY_ARGV_FLAGS.get(flag)
        if not key:
            continue
        value = inline if inline else (items[i + 1] if i + 1 < len(items) else '')
        value = value.strip().strip('"')
        # A following token that is itself a flag means the value was missing.
        if not value or (not inline and value.startswith('-')):
            continue
        try:
            if not os.path.isabs(value):
                continue
        except (OSError, ValueError):
            continue
        out[key] = os.path.normpath(value)
    return out


def detect_comfyui_folders(timeout=3) -> dict:
    """Ask the RUNNING ComfyUI which custom folders it was started with.
    NETWORK — one short GET, kept out of probe() like comfyui_runtime_info.

    Returns {} when ComfyUI is not configured, not reachable, too old to echo its
    argv (the field landed in 2025 releases), or simply started with no custom
    folder flags. An empty dict means "nothing to offer", never "use the defaults" —
    the caller leaves the manual field alone. Never raises."""
    api = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api:
        return {}
    try:
        r = requests.get(f'{api}/system_stats', timeout=timeout)
        if r.status_code != 200:
            return {}
        return parse_comfy_argv_dirs(((r.json() or {}).get('system') or {}).get('argv'))
    except Exception:
        return {}


def _detect_comfyui() -> dict:
    out = {}
    if _http_ok(f'{_COMFYUI_DEFAULT_URL}/history'):
        out['api_url'] = _COMFYUI_DEFAULT_URL
    base = _find_install_dir(('ComfyUI', 'comfyui'), _is_comfyui_dir)
    if not base:
        # portable bundle nests the app: <root>/ComfyUI_windows_portable/ComfyUI/
        portable = _find_install_dir(('ComfyUI_windows_portable',),
                                     lambda d: _is_comfyui_dir(d / 'ComfyUI'))
        if portable:
            base = str(Path(portable) / 'ComfyUI')
    if base:
        out['base_dir'] = base
    return out


def _detect_aitoolkit() -> dict:
    d = _find_install_dir(('ai-toolkit', 'ai_toolkit', 'aitoolkit'),
                          lambda p: (p / 'run.py').exists())
    return {'dir': d} if d else {}


def autodetect() -> dict:
    """Best-effort discovery of installed tools for the Setup wizard. A value under
    a reachable default port (url/api_url) is safe to auto-apply; a disk-scanned
    path (base_dir/dir) is a suggestion the UI should confirm. Never raises."""
    return {
        'ollama': _detect_ollama(),
        'comfyui': _detect_comfyui(),
        'aitoolkit': _detect_aitoolkit(),
    }


def probe(force=False) -> dict:
    global _cache, _cache_ts
    now = time.time()
    if _cache is not None and not force and (now - _cache_ts) < _CACHE_TTL:
        return copy.deepcopy(_cache)

    comfy = probe_comfyui()
    ollama = probe_ollama()
    ollama_installed = probe_ollama_installed()
    aitoolkit = probe_aitoolkit()
    gemini = probe_gemini()
    openai_ = probe_openai()
    openrouter_ = probe_openrouter()
    face_scoring = probe_face_scoring()
    masks = probe_masks()
    bank_scoring = probe_bank_scoring()
    watermark_inpaint = probe_watermark_inpaint()
    joycaption = probe_joycaption(aitoolkit)
    models = _scan_models()
    # Klein engine readiness is now honest tri-component: the graph needs the UNET
    # AND the VAE AND the text-encoder. All three gate on the RESOLVER (the exact
    # value the generate would feed each loader node), not the raw scan — so the
    # UNET check matches the vae/te ones and a model the resolver can build but the
    # old bool(models['klein']) scan structure differed on can never disagree with
    # it (picker == probe == resolver). The old unet-only check also lit the engine
    # while a generate would 409 for the missing vae/te; that 409 already names +
    # auto-downloads the gap, so a badge that flips to "not ready" here is
    # actionable. Resolvers are cheap listdir, network-free, and
    # extra_model_paths-aware. Lazy import avoids an import cycle.
    from .services import klein_edit_helper as _keh
    # Per-asset gaps (setup_installer action names still absent on disk), so the
    # Setup UI can name exactly what's missing and keep only the relevant download
    # buttons visible — judging the whole engine on the UNET alone let the step go
    # green and hid the TE/VAE buttons the moment the model landed. Disk-only
    # (network-free) and reachability-independent, so the front can separate
    # "ComfyUI unreachable" from "an asset is missing".
    klein_missing = _keh.klein_missing_assets()
    # Present-but-INVALID assets: the file EXISTS under the resolved name but is not
    # real weights — a licence-gate HTML page saved as .safetensors (the #help crash:
    # Setup green, then UNETLoader dies on "Expecting value: line 1 column 1"), a
    # truncated download, or a suspiciously tiny stub. Header-only + cached, so cheap
    # here. A REQUIRED asset that is *blocking*-invalid must NOT light the engine
    # green (advisory too_small does not gate) — an honest badge points the user at
    # the broken file instead of letting a doomed generate crash ComfyUI.
    klein_invalid = _keh.klein_invalid_assets()
    klein_blocking_invalid = any(
        i['blocking'] and i['asset'] in _keh.KLEIN_REQUIRED for i in klein_invalid)
    # Capability gap on the graph's WIDGET VALUES, not its nodes. The shipped Klein
    # workflow used to pin `scheduler: "beta57"`, a value the third-party RES4LYF
    # pack injects into ComfyUI's CORE list — so a stock install passed every asset
    # AND node check here, went green, and then refused the very first generation
    # with a raw ComfyUI 400 (reported by IndependentProcess0 on Reddit). That
    # value is gone, but the blind spot it exposed is not: this is the net for the
    # next one, and for a workflow file a user has edited themselves. Nothing is
    # substituted (a scheduler changes the render), so the gap has to be visible
    # BEFORE a batch is launched, which means here. /object_info is cached, and
    # this fails OPEN — an unreachable ComfyUI reports no gap, `reachable` already
    # says that.
    klein_unsupported_enums = _keh.klein_unsupported_enums() if comfy['ok'] else []
    klein_ready = (comfy['ok'] and not klein_unsupported_enums
                   and bool(_keh.resolve_klein_unet())
                   and bool(_keh.resolve_klein_vae())
                   and bool(_keh.resolve_klein_text_encoder())
                   and not klein_blocking_invalid)
    # Krea 2 Identity Edit — the second LOCAL engine. Readiness is honest and
    # four-part (base model + identity LoRA + text encoder + VAE) AND depends on
    # a custom-node pack, unlike Klein whose graph is core-nodes-only. Both gaps
    # are published separately so the engine card can name the RIGHT one: "install
    # the node pack" and "place the LoRA here" are different actions.
    # Disk scan = cheap listdir, network-free. The node probe is /object_info,
    # cached and fail-OPEN (unreachable ComfyUI reports no missing node here —
    # `reachable` already says that, and two red flags for one cause is noise).
    from .services import krea_edit_helper as _krh
    krea_missing = _krh.krea_missing_assets()
    krea_nodes_missing = _krh.krea_missing_nodes() if comfy['ok'] else []
    # Pack ON DISK but not yet exposed by /object_info = "restart ComfyUI", NOT
    # "install the pack". Now that the app installs the pack itself, telling
    # someone to install what they just watched install would be the whole
    # feature failing at the last inch.
    krea_nodes_installed = _krh.krea_node_pack_installed()
    # Present-but-INVALID, exactly like Klein's: the Krea base sits behind a HF
    # licence gate and the identity LoRA behind a Civitai login, so a browser
    # download without the licence/login saves the HTML gate PAGE as
    # .safetensors — present, not loadable, and today the only symptom was a raw
    # ComfyUI "Expecting value: line 1 column 1". Every Krea asset is required
    # (KREA_REQUIRED == all of them), so any blocking-invalid one keeps the
    # engine dark; the advisory too_small does not gate.
    krea_invalid = _krh.krea_invalid_assets()
    krea_blocking_invalid = any(i['blocking'] for i in krea_invalid)
    krea_ready = (comfy['ok'] and not krea_missing and not krea_nodes_missing
                  and not krea_blocking_invalid)
    # Z-Image Turbo — the third LOCAL engine (img2img, core nodes only, so NO
    # node-pack probe, like Klein). Honest tri-component readiness on the RESOLVER
    # each loader would feed, plus the present-but-invalid gate.
    from .services import zimage_edit_helper as _zih
    zimage_missing = _zih.zimage_missing_assets()
    zimage_invalid = _zih.zimage_invalid_assets()
    zimage_blocking_invalid = any(
        i['blocking'] and i['asset'] in _zih.ZIMAGE_REQUIRED for i in zimage_invalid)
    zimage_ready = (comfy['ok'] and bool(_zih.resolve_zimage_unet())
                    and bool(_zih.resolve_zimage_text_encoder())
                    and bool(_zih.resolve_zimage_vae())
                    and not zimage_blocking_invalid)
    base_dir = cfg.get('comfyui.base_dir') or ''
    comfy_dir = resolve_comfyui_base(base_dir)
    # Conscious "continue without ComfyUI" skip (Setup wizard). DERIVED, not just the
    # stored flag: a directory being configured ANNULS the skip on the spot, so the
    # flag can never mask a real error of a set-up ComfyUI — a configured install
    # always has base_dir, so `skipped` is false whenever there's something to error
    # on. The engine/studio gates below are computed independently of this and stay
    # the source of truth; `skipped` only lets the Setup step render neutral.
    comfy_skipped = bool(cfg.get('comfyui.setup_skipped')) and not base_dir

    from .services import chatgpt_oauth
    sub_status = chatgpt_oauth.status()

    from .services.face_dataset_service import MAX_FANOUT as _max_fanout

    caps = {
        'configured': cfg.is_configured(),
        # Images one generation batch may queue at once. Published so the
        # workspace can say "75 is over the limit" BEFORE the click instead of
        # letting a multi-engine run be refused after the fact — the server
        # stays the authority, the UI just mirrors the number it is told.
        'max_fanout': _max_fanout,
        'engines': {
            'nanobanana': gemini['ok'],
            'chatgpt': openai_['ok'],
            'openrouter': openrouter_['ok'],
            'klein': klein_ready,
            'krea': krea_ready,
            'zimage': zimage_ready,
        },
        'chatgpt_subscription': {
            'connected': sub_status['connected'],
            'email': sub_status['email'],
            'plan': sub_status['plan'],
            'codex_cli_detected': chatgpt_oauth.codex_auth_path().is_file(),
        },
        'comfyui': {
            'reachable': comfy['ok'],
            'api_url': cfg.get('comfyui.api_url') or '',
            'base_dir': base_dir,
            'dir_configured': bool(base_dir),
            'dir_valid': comfy_dir['valid'],       # base_dir really is a ComfyUI install
            'resolved_dir': comfy_dir['resolved'],
            # Effective "continue without ComfyUI" state: the user chose to skip AND no
            # directory is configured. Only drives the Setup step's neutral "skipped"
            # display — never the engine/studio gates below, so it cannot hide a real
            # error of a configured ComfyUI (which always has a base_dir → skipped=False).
            'skipped': comfy_skipped,
            'models': models,
            # setup_installer action names for the Klein assets NOT yet on disk
            # (subset of klein_model / klein_text_encoder / klein_vae / klein_lora).
            # Empty required-trio => the Klein engine is asset-ready.
            'klein_missing': klein_missing,
            # Widget values the shipped Klein graph needs that THIS ComfyUI doesn't
            # offer: [{node_id, class_type, input, value, pack, url}]. Empty on a
            # capable install AND on an unreachable one (fail-open).
            'klein_unsupported_enums': klein_unsupported_enums,
            # Krea 2 Edit gaps, kept apart from Klein's: asset KEYS not on disk
            # (krea_edit_helper.KREA_ASSETS) and the custom-node class_types this
            # ComfyUI doesn't expose. Empty + empty => the engine is ready.
            'krea_missing': krea_missing,
            'krea_nodes_missing': krea_nodes_missing,
            'krea_nodes_installed': krea_nodes_installed,
            # Krea assets PRESENT on disk but not real, loadable weights — same
            # [{asset, filename, verdict, blocking, reason}] shape as
            # klein_invalid, so one banner covers both engines.
            'krea_invalid': krea_invalid,
            'zimage_missing': zimage_missing,
            'zimage_invalid': zimage_invalid,
            # Klein assets PRESENT on disk but not real, loadable weights:
            # [{asset, filename, verdict, blocking, reason}]. Distinct from
            # klein_missing (the file exists, it just can't load) — drives the Setup
            # "present but INVALID: <asset> (<reason>)" line and the diagnostic, and
            # a blocking-invalid required asset also keeps engines.klein dark above.
            'klein_invalid': klein_invalid,
        },
        'ollama': {
            'reachable': ollama['ok'],
            # Installed = binary on disk, even when the server is stopped. The UI
            # reads (installed, reachable) as three states: not installed /
            # installed-but-stopped (→ "Start Ollama" button) / running.
            'installed': ollama_installed['ok'],
            'binary_path': ollama_installed['binary_path'],   # local-only; drives the Start route
            'url': cfg.get('ollama.url') or '',
            'vision_model': cfg.get('ollama.vision_model') or '',
            'vision_model_ready': probe_ollama_model(reachable=ollama['ok'])['ok'],
        },
        'aitoolkit': {
            'configured': bool(cfg.get('aitoolkit.dir')),
            'valid': aitoolkit['ok'],
            # Kept apart so Setup can report what it ACTUALLY found: "this isn't
            # an ai-toolkit checkout" and "this checkout has no interpreter we
            # can see" are different problems with different fixes.
            'dir_valid': bool(aitoolkit.get('has_run')),
            'python_candidates': list(aitoolkit.get('python_candidates') or []),
        },
        'cloud_training': bool(cfg.secret('VAST_API_KEY')),
        # Publish-to-HF is gated purely on the HF_TOKEN secret being present (the
        # write-scope check is a live preflight at publish time, not here — probe()
        # must stay network-free). The ⋯ More menu entry keys off this.
        'hf_publish': bool(cfg.secret('HF_TOKEN')),
        'captioners': {
            # Honest: the ai-toolkit venv must actually import the JoyCaption deps,
            # not merely have the script on disk (issue #6). `detail` carries the
            # exact pip command when it can't, so the UI/error can name the fix.
            'joycaption': joycaption['ok'],
            'joycaption_detail': joycaption['detail'],
            'ollama': ollama['ok'],
        },
        'face_scoring': face_scoring['ok'],
        'masks': masks['ok'],
        # Bank scoring extra (CLIP aesthetic + NSFW + style clustering). Gates the
        # bank's "Score (aesthetic · NSFW · style)" button; False → install hint.
        'bank_scoring': bank_scoring['ok'],
        # Lets the front adapt the watermark Clean tooltip: when False, Clean is
        # crop-only (LaMa-routed watermarks are skipped with an install hint).
        'watermark_inpaint': watermark_inpaint['ok'],
        # Klein-inpaint (V2, quality) readiness = same as the Klein engine (ComfyUI
        # reachable + Klein models on disk). The custom-node preflight is a clean-time
        # 409. Greys the batch's "Klein (quality)" option when False.
        'watermark_klein': klein_ready,
        # Persisted "allow automatic crop" preference (Settings ▸ Watermark inpainting).
        # The batch Clean bar reads it here to seed/reflect its inline toggle and the
        # review lightbox uses it as the per-image crop-vs-inpaint default; when False,
        # auto-routing repaints border marks instead of cropping them.
        'watermark_allow_crop': bool(cfg.get('watermark.allow_crop')),
        'python': python_ml_status(),
        'scrape_deps': probe_scrape_deps()['ok'],
        'training_visible': aitoolkit['ok'] or bool(cfg.secret('VAST_API_KEY')),
        'studio_visible': comfy['ok'],
    }

    _cache, _cache_ts = caps, now
    return copy.deepcopy(caps)
