from app import setup_installer as si


def test_zimage_download_specs_public_and_renamed_vae():
    d = si._ZIMAGE_DOWNLOADS
    assert set(d) == {'zimage_model', 'zimage_text_encoder', 'zimage_vae'}
    assert d['zimage_model']['url'].startswith('https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/')
    for spec in d.values():
        assert spec['gated'] is False
    assert si._ZIMAGE_DOWNLOADS['zimage_vae']['dest'][-1] == 'z_image_ae.safetensors'


def test_actions_registered_with_workers():
    for a in ('zimage_model', 'zimage_text_encoder', 'zimage_vae'):
        assert a in si.INSTALL_ACTIONS
        assert a in si._WORKERS


def test_install_all_plan_excludes_zimage():
    # Z-Image is a SECONDARY engine, so — like Krea — it is deliberately absent
    # from "Install everything" (which runs unattended and would otherwise pull a
    # ~15 GB engine nobody asked for). It installs on intent instead: the per-asset
    # Setup buttons and the auto-start when a user picks the engine and generates.
    caps = {'comfyui': {'dir_valid': True, 'klein_missing': [],
                        'zimage_missing': ['zimage_model', 'zimage_text_encoder',
                                           'zimage_vae']}}
    plan = si.install_all_plan(caps)
    assert not any(a.startswith('zimage_') for a in plan)
