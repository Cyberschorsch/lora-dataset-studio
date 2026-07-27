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
