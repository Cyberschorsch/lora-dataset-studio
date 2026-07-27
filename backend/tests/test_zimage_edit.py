from app.services import zimage_edit_helper as z


def _graph():
    return z.build_workflow(
        'ref.png', 'a portrait', unet='z image/z_image_turbo_bf16.safetensors',
        clip='qwen_3_4b.safetensors', vae='z_image_ae.safetensors',
        seed=42, steps=8, denoise=0.65, filename_prefix='u_DatasetZImage_abcd1234')


def test_loaders_wired():
    g = _graph()
    assert g['1']['class_type'] == 'UNETLoader'
    assert g['1']['inputs']['unet_name'] == 'z image/z_image_turbo_bf16.safetensors'
    assert g['2']['class_type'] == 'CLIPLoader'
    assert g['2']['inputs']['type'] == 'lumina2'
    assert g['3']['inputs']['vae_name'] == 'z_image_ae.safetensors'


def test_is_img2img_reference_latent_feeds_sampler():
    g = _graph()
    load = next(k for k, n in g.items() if n['class_type'] == 'LoadImage')
    enc = next(k for k, n in g.items() if n['class_type'] == 'VAEEncode')
    samp = next(k for k, n in g.items() if n['class_type'] == 'SamplerCustomAdvanced')
    assert g[enc]['inputs']['pixels'] == [load, 0]
    assert g[samp]['inputs']['latent_image'] == [enc, 0]
    assert not any(n['class_type'] == 'EmptySD3LatentImage' for n in g.values())


def test_denoise_steps_and_pinned_cfg():
    g = _graph()
    sched = next(n for n in g.values() if n['class_type'] == 'BasicScheduler')
    guider = next(n for n in g.values() if n['class_type'] == 'CFGGuider')
    assert sched['inputs']['denoise'] == 0.65
    assert sched['inputs']['steps'] == 8
    assert guider['inputs']['cfg'] == 1.0


def test_seed_and_unique_prefix():
    g = _graph()
    noise = next(n for n in g.values() if n['class_type'] == 'RandomNoise')
    save = next(n for n in g.values() if n['class_type'] == 'SaveImage')
    assert noise['inputs']['noise_seed'] == 42
    assert save['inputs']['filename_prefix'] == 'u_DatasetZImage_abcd1234'


def test_fit_output_size_caps_and_snaps():
    w, h = z.fit_output_size(4000, 3000, max_mp=1.5)
    assert w % 16 == 0 and h % 16 == 0
    assert w * h <= 1.5 * 1_000_000 * 1.05
    assert z.fit_output_size(512, 512, max_mp=1.5) == (512, 512)


def test_enqueue_builds_and_queues(tmp_path, monkeypatch):
    root = tmp_path
    from PIL import Image
    src_dir = root / 'out'; src_dir.mkdir()
    inp_dir = root / 'in'; inp_dir.mkdir()
    ref = src_dir / 'ref.png'
    Image.new('RGB', (800, 1200), 'white').save(ref)

    monkeypatch.setattr(z, 'resolve_zimage_unet', lambda selected=None: 'z image/z_image_turbo_bf16.safetensors')
    monkeypatch.setattr(z, 'resolve_zimage_text_encoder', lambda: 'qwen_3_4b.safetensors')
    monkeypatch.setattr(z, 'resolve_zimage_vae', lambda: 'z_image_ae.safetensors')
    monkeypatch.setattr(z, 'preflight', lambda: None)
    monkeypatch.setattr(z.cfg, 'comfyui_dir', lambda kind: str(inp_dir) if kind == 'input' else str(src_dir))
    monkeypatch.setattr(z.cfg, 'get', lambda *a, **k: None)

    captured = {}
    monkeypatch.setattr(z.queue_manager, 'add_job', lambda **kw: captured.update(kw))
    job_id = z.enqueue_zimage_edit(user_id='7', source_filename='ref.png',
                                   edit_prompt='a portrait', source_path=str(ref))
    assert job_id
    assert any(p.name.startswith('zimage_source_') for p in inp_dir.iterdir())
    wf = captured['workflow_data']
    assert wf['1']['class_type'] == 'UNETLoader'
    assert captured['job_type'] == 'image'
