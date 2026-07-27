"""Shot geometry for the Z-Image graph family.

The engine's reference is `ref_filename`, the SQUARE head crop, and before the
graph family the output canvas copied that square's aspect — so a 'body' shot
could only ever come back a head. These tests pin the planner that fixes it.
"""
import pytest

from app.services import zimage_edit_helper as z


ASPECTS = ('1:1', '3:4', '4:3', '16:9', '9:16')


def test_aspect_size_snaps_and_fits_budget():
    for a in ASPECTS:
        w, h = z.aspect_size(a)
        assert w % 16 == 0 and h % 16 == 0, a
        assert w * h <= 1.5 * 1_000_000 * 1.05, a
        aw, ah = (float(x) for x in a.split(':'))
        assert abs((w / h) - (aw / ah)) < 0.03, a


def test_aspect_size_falls_back_to_square():
    for bad in (None, '', 'nonsense', '0:3', '3:0', 5):
        w, h = z.aspect_size(bad)
        assert w == h, bad


def test_mode_selection_per_framing():
    def mode(framing, aspect):
        return z.plan_shot(framing, aspect, (1024, 1024), denoise=0.75)['mode']

    assert mode('face', '1:1') == 'close'
    assert mode('bust', '3:4') == 'restage'
    assert mode('body', '3:4') == 'restage'
    # Legacy rows have no framing at all -> the pre-graph-family behaviour.
    assert mode(None, None) == 'close'
    assert mode('unknown', None) == 'close'


def test_face_shot_with_wide_override_is_restaged():
    """A 'face' entry carrying a 16:9 override is a scene shot wearing a close-up's
    tag — letterboxing a head into it is exactly the old bug."""
    assert z.plan_shot('face', '16:9', (1024, 1024), denoise=0.75)['mode'] == 'restage'


def test_back_never_pastes_a_head():
    """A frontal head composited into a back shot is a head-on-backwards artifact."""
    no_ref = z.plan_shot('back', '3:4', (1024, 1024), denoise=0.75)
    assert no_ref['mode'] == 'backdrop'
    assert no_ref['paste'] is None and no_ref['ellipse'] is None

    with_full = z.plan_shot('back', '3:4', (1024, 1024), denoise=0.75,
                            has_full_frame=True)
    assert with_full['mode'] == 'close'          # bias from the full frame instead
    assert with_full['source'] == 'original'
    assert with_full['paste'] is None


@pytest.mark.parametrize('framing', ('bust', 'body'))
@pytest.mark.parametrize('aspect', ASPECTS)
def test_paste_and_ellipse_stay_inside_the_canvas(framing, aspect):
    p = z.plan_shot(framing, aspect, (1024, 1024), denoise=0.75)
    w, h = p['canvas']
    px, py, side = p['paste']
    assert 0 <= px and px + side <= w
    assert 0 <= py and py + side <= h
    cx, cy, rx, ry = p['ellipse']
    assert 0 <= cx - rx and cx + rx <= w
    assert 0 <= cy - ry and cy + ry <= h


def test_bust_head_is_bigger_than_a_body_head():
    b = z.plan_shot('bust', '3:4', (1024, 1024), denoise=0.75)
    y = z.plan_shot('body', '3:4', (1024, 1024), denoise=0.75)
    assert b['paste'][2] > y['paste'][2]
    # A body head is far smaller in latent terms, so it is held tighter.
    assert y['keep'] < b['keep']


def test_restage_always_samples_at_full_denoise():
    """Composition freedom comes from the CANVAS, not from a partial denoise — the
    identity is protected by the mask instead. This is the Krea analog."""
    for framing in ('bust', 'body'):
        assert z.plan_shot(framing, '3:4', (1024, 1024), denoise=0.75)['denoise'] == 1.0


def test_denoise_bias_keeps_the_dial_monotone():
    assert z.denoise_for('face', 0.75) < 0.75          # face shots stay closer
    assert z.denoise_for('face', 0.5) < z.denoise_for('face', 0.9)
    assert z.denoise_for(None, 0.75) == 0.75           # legacy path unchanged


def test_keep_value_monotone_and_clamped():
    assert z.keep_value(0.40) == 0.0
    assert z.keep_value(0.10) == 0.0
    assert z.keep_value(0.75) > z.keep_value(0.60)
    assert z.keep_value(1.0) <= 0.60


def test_body_source_setting_opts_into_the_full_frame():
    crop = z.plan_shot('body', '3:4', (1024, 1024), denoise=0.75,
                       body_source='crop', has_full_frame=True)
    orig = z.plan_shot('body', '3:4', (1024, 1024), denoise=0.75,
                       body_source='original', has_full_frame=True)
    none = z.plan_shot('body', '3:4', (1024, 1024), denoise=0.75,
                       body_source='original', has_full_frame=False)
    assert crop['source'] == 'crop'          # default = today's safe behaviour
    assert orig['source'] == 'original'
    assert none['source'] == 'crop'          # opted in, but nothing to opt into


def test_plan_shot_is_pure(monkeypatch):
    """No config read, no disk — the whole point of splitting it out."""
    def boom(*a, **k):
        raise AssertionError('plan_shot must not read config')
    monkeypatch.setattr(z.cfg, 'get', boom)
    for framing in ('face', 'bust', 'body', 'back', None):
        z.plan_shot(framing, '3:4', (1024, 1024), denoise=0.75)


def test_close_shots_honour_the_aspect_without_upscaling():
    """Honouring the shot's aspect must not resurrect the upscale fit_output_size
    deliberately refused — a small crop interpolated up is invented detail the LoRA
    would then learn."""
    small = z.plan_shot('face', '1:1', (512, 512), denoise=0.75)['canvas']
    assert small[0] * small[1] <= 512 * 512 * 1.1, small
    big = z.plan_shot('face', '1:1', (2048, 2048), denoise=0.75)['canvas']
    assert big[0] * big[1] <= 1.5 * 1_000_000 * 1.05
    assert big[0] == big[1]


def test_fit_cover_crops_instead_of_squashing():
    from PIL import Image
    out = z.fit_cover(Image.new('RGB', (1000, 1000), 'red'), (600, 900))
    assert out.size == (600, 900)
    # A stretch would keep every source pixel; a cover-crop discards the overflow
    # while the scale stays uniform on both axes.
    tall = z.fit_cover(Image.new('RGB', (1000, 500), 'red'), (400, 800))
    assert tall.size == (400, 800)
