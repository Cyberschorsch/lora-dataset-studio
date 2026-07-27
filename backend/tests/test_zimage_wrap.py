from app.services import face_variations as fv

# Z-Image is a plain BASE text-to-image model driven by img2img; it never receives
# the reference as semantic conditioning. So its prompt must NOT reuse the Klein/Krea
# instruction-edit wrapper (which addresses "the reference image" and commands a
# "restage") — that language is dead weight to a base model and buried the real
# pose/framing cue, so every shot came back a near-copy of the reference.

_ARGS = dict(nsfw=False, framing='face', suffix='', subject_type='human',
             label='Face front, neutral')


def test_zimage_wrap_diverges_from_klein():
    # The two engines are no longer the same prompt: Z-Image drops the edit language.
    assert (fv.wrap_variation_zimage('a portrait', **_ARGS)
            != fv.wrap_variation_klein('a portrait', **_ARGS))


def test_zimage_wrap_has_no_edit_instruction_language():
    p = fv.wrap_variation_zimage('a portrait', **_ARGS).lower()
    # None of the Kontext/edit-model phrasing a base model cannot act on.
    for banned in ('reference image', 'restage', 'same person',
                   'facial identity', 'do not copy'):
        assert banned not in p, f'base-model prompt still carries edit language: {banned!r}'


def test_klein_wrap_still_carries_edit_language():
    # Contrast: the edit engines DO keep the instruction wrapper (regression guard).
    assert 'reference image' in fv.wrap_variation_klein('a portrait', **_ARGS).lower()


def test_zimage_wrap_keeps_descriptive_body_and_tail():
    p = fv.wrap_variation_zimage('a portrait', **_ARGS)
    assert 'a portrait' in p                             # the creative body survives
    assert 'Close-up head-and-shoulders portrait' in p   # per-framing detail block
    assert 'Professional realistic photograph' in p      # SFW render tail
