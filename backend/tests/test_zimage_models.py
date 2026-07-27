import os
from app.services import zimage_edit_helper as z


def _mk(root, comfy_type, *rel_files):
    for rel in rel_files:
        p = os.path.join(root, 'models', comfy_type, *rel.split('/'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh:
            fh.write(b'\x00' * 16)


def test_unet_resolves_with_subfolder_prefix(tmp_path, monkeypatch):
    root = str(tmp_path)
    _mk(root, 'diffusion_models', 'z image/z_image_turbo_bf16.safetensors')

    def roots(comfy_type):
        return [os.path.join(root, 'models', comfy_type)]
    monkeypatch.setattr(z.comfy_model_paths, 'search_roots', roots)
    monkeypatch.setattr(z.cfg, 'get', lambda *a, **k: None)

    assert z.resolve_zimage_unet() == os.path.join('z image', 'z_image_turbo_bf16.safetensors')


def test_vae_accepts_both_downloaded_and_manual_names(tmp_path, monkeypatch):
    root = str(tmp_path)

    def roots(comfy_type):
        return [os.path.join(root, 'models', comfy_type)]
    monkeypatch.setattr(z.comfy_model_paths, 'search_roots', roots)
    _mk(root, 'vae', 'z ae.safetensors')
    assert z.resolve_zimage_vae() == 'z ae.safetensors'


def test_vae_does_not_match_flux_ae(tmp_path, monkeypatch):
    root = str(tmp_path)

    def roots(comfy_type):
        return [os.path.join(root, 'models', comfy_type)]
    monkeypatch.setattr(z.comfy_model_paths, 'search_roots', roots)
    _mk(root, 'vae', 'ae.safetensors')
    _mk(root, 'vae', 'flux2-vae.safetensors')
    assert z.resolve_zimage_vae() is None


def test_text_encoder_narrow_qwen_3_4b(tmp_path, monkeypatch):
    root = str(tmp_path)

    def roots(comfy_type):
        return [os.path.join(root, 'models', comfy_type)]
    monkeypatch.setattr(z.comfy_model_paths, 'search_roots', roots)
    _mk(root, 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
    _mk(root, 'text_encoders', 'qwen3vl_4b_fp8_scaled.safetensors')
    assert z.resolve_zimage_text_encoder() is None
    _mk(root, 'text_encoders', 'qwen_3_4b.safetensors')
    assert z.resolve_zimage_text_encoder() == 'qwen_3_4b.safetensors'


def test_missing_assets_lists_absent_keys(tmp_path, monkeypatch):
    def roots(comfy_type):
        return [os.path.join(str(tmp_path), 'models', comfy_type)]
    monkeypatch.setattr(z.comfy_model_paths, 'search_roots', roots)
    monkeypatch.setattr(z.cfg, 'get', lambda *a, **k: None)
    assert z.zimage_missing_assets() == ['zimage_model', 'zimage_text_encoder', 'zimage_vae']
