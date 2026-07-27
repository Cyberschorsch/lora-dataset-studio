from app.services import face_variations as fv


def test_zimage_wrap_matches_klein():
    args = dict(nsfw=False, framing='face', suffix='', subject_type='human', label='Face front, neutral')
    assert fv.wrap_variation_zimage('a portrait', **args) == fv.wrap_variation_klein('a portrait', **args)
