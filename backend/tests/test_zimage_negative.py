"""The Z-Image BASE negative prompt.

Turbo runs at cfg 1.0, where the negative branch has no effect at all — this text
only ever reaches Base. It lives in face_variations because it is prompt text that
varies by subject type.
"""
from app.services import face_variations as fv


def test_human_negative_covers_the_usual_photo_failures():
    n = fv.zimage_negative('human')
    for token in ('watermark', 'extra fingers', 'blurry', 'text'):
        assert token in n


def test_drawn_subjects_keep_their_medium():
    """'cartoon, illustration, 3d render' as a negative would fight the render tail
    an anime subject explicitly asks for."""
    anime = fv.zimage_negative('anime')
    human = fv.zimage_negative('human')
    for token in ('cartoon', 'illustration', '3d render', 'painting'):
        assert token not in anime, token
        assert token in human, token
    assert 'watermark' in anime          # the medium-neutral half still applies


def test_unknown_subject_falls_back_to_human():
    assert fv.zimage_negative('nonsense') == fv.zimage_negative('human')
    assert fv.zimage_negative(None) == fv.zimage_negative('human')


def test_setting_replaces_it_wholesale(monkeypatch):
    from app import config as cfg
    monkeypatch.setattr(cfg, 'get',
                        lambda k, *a, **kw: 'just this' if k == 'zimage.base_negative' else None)
    assert fv.zimage_negative('human') == 'just this'


def test_blank_setting_means_default(monkeypatch):
    from app import config as cfg
    monkeypatch.setattr(cfg, 'get',
                        lambda k, *a, **kw: '   ' if k == 'zimage.base_negative' else None)
    assert 'watermark' in fv.zimage_negative('human')
