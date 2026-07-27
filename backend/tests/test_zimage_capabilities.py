from app import capabilities


def test_engine_dark_when_assets_missing(monkeypatch):
    from app.services import zimage_edit_helper as z
    monkeypatch.setattr(z, 'resolve_zimage_unet', lambda selected=None: None)
    monkeypatch.setattr(z, 'resolve_zimage_text_encoder', lambda: None)
    monkeypatch.setattr(z, 'resolve_zimage_vae', lambda: None)
    monkeypatch.setattr(z, 'zimage_missing_assets', lambda: ['zimage_model', 'zimage_text_encoder', 'zimage_vae'])
    monkeypatch.setattr(z, 'zimage_invalid_assets', lambda: [])
    caps = capabilities.probe(force=True)
    assert caps['engines']['zimage'] is False
    assert 'zimage_model' in caps['comfyui']['zimage_missing']


def test_engine_ready_when_all_present(monkeypatch):
    from app.services import zimage_edit_helper as z
    monkeypatch.setattr(capabilities, 'probe_comfyui', lambda: {'ok': True})
    monkeypatch.setattr(z, 'resolve_zimage_unet', lambda selected=None: 'z image/u.safetensors')
    monkeypatch.setattr(z, 'resolve_zimage_text_encoder', lambda: 'qwen_3_4b.safetensors')
    monkeypatch.setattr(z, 'resolve_zimage_vae', lambda: 'z_image_ae.safetensors')
    monkeypatch.setattr(z, 'zimage_missing_assets', lambda: [])
    monkeypatch.setattr(z, 'zimage_invalid_assets', lambda: [])
    caps = capabilities.probe(force=True)
    assert caps['engines']['zimage'] is True
