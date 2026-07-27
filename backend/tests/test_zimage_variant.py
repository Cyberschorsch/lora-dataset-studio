"""Turbo vs Base.

Turbo is guidance-distilled: cfg is pinned to 1.0 and the negative branch is inert.
Base is not distilled, so it takes a real cfg, more steps and a real negative.
Detection defaults to TURBO on purpose — Base at cfg 1.0/8 steps is merely soft,
Turbo at cfg 4.0/28 steps is burnt garbage.
"""
from PIL import Image

from app.services import zimage_edit_helper as z


def _cfg(monkeypatch, values):
    monkeypatch.setattr(z.cfg, 'get', lambda k, *a, **kw: values.get(k))


def test_detects_turbo_and_base_from_the_filename(monkeypatch):
    _cfg(monkeypatch, {})
    assert z.zimage_variant('z image/z_image_turbo_bf16.safetensors') == 'turbo'
    assert z.zimage_variant('z image/z_image_bf16.safetensors') == 'base'
    assert z.zimage_variant('z image/z_image_base_fp8.safetensors') == 'base'


def test_unknown_filename_fails_safe_to_turbo(monkeypatch):
    _cfg(monkeypatch, {})
    assert z.zimage_variant('z image/mystery.safetensors') == 'turbo'
    assert z.zimage_variant(None) == 'turbo'


def test_setting_overrides_the_filename(monkeypatch):
    _cfg(monkeypatch, {'zimage.variant': 'base'})
    assert z.zimage_variant('z image/z_image_turbo_bf16.safetensors') == 'base'
    _cfg(monkeypatch, {'zimage.variant': 'turbo'})
    assert z.zimage_variant('z image/z_image_bf16.safetensors') == 'turbo'
    _cfg(monkeypatch, {'zimage.variant': 'auto'})
    assert z.zimage_variant('z image/z_image_bf16.safetensors') == 'base'


def test_variant_base_also_steers_which_file_is_resolved(monkeypatch):
    """Asking for Base must not keep loading the Turbo file: a distilled model run
    at cfg 4.0 / 28 steps is burnt output, not a soft one."""
    both = [('z image', ['z_image_bf16.safetensors', 'z_image_turbo_bf16.safetensors'])]
    monkeypatch.setattr(z, '_zimage_unet_folders', lambda: both)

    _cfg(monkeypatch, {})
    assert z.resolve_zimage_unet().endswith('z_image_turbo_bf16.safetensors')
    _cfg(monkeypatch, {'zimage.variant': 'base'})
    assert z.resolve_zimage_unet().endswith('z_image_bf16.safetensors')
    # An explicit file pin still outranks the variant.
    _cfg(monkeypatch, {'zimage.variant': 'base',
                       'zimage.base_model': 'z_image_turbo_bf16.safetensors'})
    assert z.resolve_zimage_unet().endswith('z_image_turbo_bf16.safetensors')


def test_variant_base_without_a_base_build_still_resolves(monkeypatch):
    """No non-turbo build on disk -> fall back rather than return None and block."""
    monkeypatch.setattr(z, '_zimage_unet_folders',
                        lambda: [('z image', ['z_image_turbo_bf16.safetensors'])])
    _cfg(monkeypatch, {'zimage.variant': 'base'})
    assert z.resolve_zimage_unet().endswith('z_image_turbo_bf16.safetensors')


def _enqueue(tmp_path, monkeypatch, unet, settings, **kw):
    src_dir = tmp_path / 'out'; src_dir.mkdir(exist_ok=True)
    inp_dir = tmp_path / 'in'; inp_dir.mkdir(exist_ok=True)
    ref = src_dir / 'ref.png'
    Image.new('RGB', (800, 800), 'white').save(ref)

    monkeypatch.setattr(z, 'resolve_zimage_unet', lambda selected=None: unet)
    monkeypatch.setattr(z, 'resolve_zimage_text_encoder', lambda: 'qwen_3_4b.safetensors')
    monkeypatch.setattr(z, 'resolve_zimage_vae', lambda: 'z_image_ae.safetensors')
    monkeypatch.setattr(z, 'preflight', lambda: None)
    monkeypatch.setattr(z.cfg, 'comfyui_dir',
                        lambda kind: str(inp_dir) if kind == 'input' else str(src_dir))
    _cfg(monkeypatch, settings)
    captured = {}
    monkeypatch.setattr(z.queue_manager, 'add_job', lambda **kw2: captured.update(kw2))
    z.enqueue_zimage_edit(user_id='7', source_filename='ref.png',
                          edit_prompt='a portrait', source_path=str(ref), **kw)
    return captured['workflow_data'], inp_dir


def test_turbo_pins_cfg_and_drops_the_negative(tmp_path, monkeypatch):
    wf, _ = _enqueue(tmp_path, monkeypatch, 'z image/z_image_turbo_bf16.safetensors',
                     {}, negative_prompt='blurry, watermark')
    assert wf['10']['inputs']['cfg'] == 1.0
    assert wf['5']['inputs']['text'] == ''
    assert wf['8']['inputs']['steps'] == 8


def test_base_uses_real_cfg_steps_and_negative(tmp_path, monkeypatch):
    wf, _ = _enqueue(tmp_path, monkeypatch, 'z image/z_image_bf16.safetensors',
                     {}, negative_prompt='blurry, watermark')
    assert wf['10']['inputs']['cfg'] == 4.0
    assert wf['8']['inputs']['steps'] == 28
    assert wf['5']['inputs']['text'] == 'blurry, watermark'


def test_base_knobs_are_configurable(tmp_path, monkeypatch):
    wf, _ = _enqueue(tmp_path, monkeypatch, 'z image/z_image_bf16.safetensors',
                     {'zimage.base_cfg': 6.5, 'zimage.base_steps': 40},
                     negative_prompt='x')
    assert wf['10']['inputs']['cfg'] == 6.5
    assert wf['8']['inputs']['steps'] == 40


def test_restage_shot_writes_a_canvas_and_a_mask(tmp_path, monkeypatch):
    wf, inp_dir = _enqueue(tmp_path, monkeypatch,
                           'z image/z_image_turbo_bf16.safetensors', {},
                           framing='body', aspect='3:4')
    names = sorted(p.name for p in inp_dir.iterdir())
    assert any(n.startswith('zimage_source_') for n in names), names
    assert any(n.startswith('zimage_mask_') for n in names), names
    assert wf['15']['class_type'] == 'LoadImageMask'
    assert wf['8']['inputs']['denoise'] == 1.0


def test_legacy_call_without_framing_is_the_old_graph(tmp_path, monkeypatch):
    wf, inp_dir = _enqueue(tmp_path, monkeypatch,
                           'z image/z_image_turbo_bf16.safetensors', {})
    assert not any(n.startswith('zimage_mask_') for n in
                   (p.name for p in inp_dir.iterdir()))
    assert '15' not in wf and '16' not in wf and '17' not in wf
    assert wf['8']['inputs']['denoise'] == 0.75
