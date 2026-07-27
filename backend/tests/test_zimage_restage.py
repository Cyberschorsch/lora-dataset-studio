"""The three graph shapes build_workflow can emit.

The mask convention is the one that bites: ComfyUI's KSamplerX0Inpaint computes
`latent_mask = 1 - denoise_mask`, so mask 1 = REGENERATE and mask 0 = preserve
verbatim. The keep-mask is therefore white everywhere except the head, and it is
read on the RED channel because LoadImageMask's alpha path inverts.
"""
import pytest

from app.services import zimage_edit_helper as z


def _wf(**kw):
    return z.build_workflow(
        'canvas.png', 'a portrait', unet='z image/z_image_turbo_bf16.safetensors',
        clip='qwen_3_4b.safetensors', vae='z_image_ae.safetensors',
        seed=42, filename_prefix='u_DatasetZImage_abcd1234', **kw)


def _by_class(g, cls):
    return next((k for k, n in g.items() if n['class_type'] == cls), None)


def test_restage_wires_the_mask_between_encode_and_sampler():
    g = _wf(mask_image='mask.png', denoise=1.0)
    lm = _by_class(g, 'LoadImageMask')
    snm = _by_class(g, 'SetLatentNoiseMask')
    enc = _by_class(g, 'VAEEncode')
    samp = _by_class(g, 'SamplerCustomAdvanced')
    assert lm and snm
    assert g[lm]['inputs']['image'] == 'mask.png'
    # RED, never alpha: LoadImageMask inverts the alpha channel (1 - alpha).
    assert g[lm]['inputs']['channel'] == 'red'
    assert g[snm]['inputs']['samples'] == [enc, 0]
    assert g[snm]['inputs']['mask'] == [lm, 0]
    assert g[samp]['inputs']['latent_image'] == [snm, 0]


def test_backdrop_has_no_reference_at_all():
    g = _wf(empty_latent=(912, 1632))
    empty = _by_class(g, 'EmptySD3LatentImage')
    samp = _by_class(g, 'SamplerCustomAdvanced')
    assert empty
    assert g[empty]['inputs'] == {'width': 912, 'height': 1632, 'batch_size': 1}
    assert g[samp]['inputs']['latent_image'] == [empty, 0]
    assert _by_class(g, 'LoadImage') is None
    assert _by_class(g, 'VAEEncode') is None
    assert _by_class(g, 'LoadImageMask') is None


def test_close_is_unchanged_by_default():
    """No mask, no empty latent -> the pre-graph-family img2img graph."""
    g = _wf()
    assert _by_class(g, 'LoadImageMask') is None
    assert _by_class(g, 'SetLatentNoiseMask') is None
    assert _by_class(g, 'EmptySD3LatentImage') is None
    enc = _by_class(g, 'VAEEncode')
    assert g[_by_class(g, 'SamplerCustomAdvanced')]['inputs']['latent_image'] == [enc, 0]


def test_mask_and_empty_latent_are_mutually_exclusive():
    with pytest.raises(ValueError):
        _wf(mask_image='mask.png', empty_latent=(512, 512))


def test_cfg_and_negative_default_to_the_turbo_contract():
    g = _wf()
    assert g['10']['inputs']['cfg'] == 1.0
    assert g['5']['inputs']['text'] == ''


def test_base_takes_a_real_cfg_and_negative():
    g = _wf(cfg_scale=4.0, negative_prompt='blurry, watermark')
    assert g['10']['inputs']['cfg'] == 4.0
    assert g['5']['inputs']['text'] == 'blurry, watermark'


def test_build_workflow_reads_no_config(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('build_workflow must stay pure')
    monkeypatch.setattr(z.cfg, 'get', boom)
    _wf(mask_image='mask.png')
    _wf(empty_latent=(512, 512))
    _wf()
