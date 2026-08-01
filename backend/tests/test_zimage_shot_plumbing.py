"""The shot's geometry has to REACH the engine.

Before the graph family, generate_variations_zimage passed `framing` only to the
prompt wrapper and to the DB row — the helper never saw it, so every shot rendered
at the square head crop's own aspect. These tests lock the wiring on both the
fan-out and the regenerate path.
"""
import io
import os

from PIL import Image

from app.config import LOCAL_USER


def _png(color=(255, 0, 0), size=(64, 64)):
    buf = io.BytesIO(); Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


def _ds_with_ref(svc, name='Lola', trigger='lola', with_original=False, **kwargs):
    ds = svc.create_dataset(LOCAL_USER, name, trigger, **kwargs)
    d = svc._dataset_dir(ds.id); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    if with_original:
        with open(os.path.join(d, 'full.webp'), 'wb') as fh:
            fh.write(_png((0, 0, 255), (600, 900)))
        ds.ref_original_filename = 'full.webp'
    svc.db.session.commit()
    return ds


def _capture(monkeypatch):
    from app.services import zimage_edit_helper as zih
    got = []
    monkeypatch.setattr(zih, 'preflight', lambda: None)
    monkeypatch.setattr(zih, 'enqueue_zimage_edit',
                        lambda **k: got.append(k) or f'job-{len(got)}')
    return got


def test_fanout_forwards_framing_and_the_catalog_aspect(app, monkeypatch):
    from app.services import face_dataset_service as svc
    got = _capture(monkeypatch)
    with app.app_context():
        ds = _ds_with_ref(svc)
        svc.generate_variations_zimage(
            LOCAL_USER, ds.id,
            [{'label': 'Face front, neutral', 'framing': 'face',
              'prompt': 'close-up portrait'},
             {'label': 'Body standing, front', 'framing': 'body',
              'prompt': 'full body shot'}],
            1)
    assert got[0]['framing'] == 'face'
    assert got[1]['framing'] == 'body'
    # The catalog resolves each shot's aspect server-side; both must be non-blank.
    assert got[0]['aspect'] and got[1]['aspect']
    assert ':' in got[0]['aspect']


def test_fanout_passes_the_full_frame_only_when_one_exists(app, monkeypatch):
    from app.services import face_dataset_service as svc
    got = _capture(monkeypatch)
    with app.app_context():
        bare = _ds_with_ref(svc, name='Bare', trigger='bare')
        svc.generate_variations_zimage(
            LOCAL_USER, bare.id,
            [{'label': 'a', 'framing': 'back', 'prompt': 'from behind'}], 1)
        full = _ds_with_ref(svc, name='Full', trigger='full', with_original=True)
        svc.generate_variations_zimage(
            LOCAL_USER, full.id,
            [{'label': 'b', 'framing': 'back', 'prompt': 'from behind'}], 1)
    assert got[0]['full_frame_path'] is None
    assert got[1]['full_frame_path'] and got[1]['full_frame_path'].endswith('full.webp')


def test_fanout_sends_a_negative_prompt(app, monkeypatch):
    """Inert on Turbo (cfg 1.0), real on Base — the helper decides which."""
    from app.services import face_dataset_service as svc
    got = _capture(monkeypatch)
    with app.app_context():
        ds = _ds_with_ref(svc)
        svc.generate_variations_zimage(
            LOCAL_USER, ds.id,
            [{'label': 'a', 'framing': 'face', 'prompt': 'close-up'}], 1)
    assert 'watermark' in got[0]['negative_prompt']


def test_regenerate_forwards_the_same_shot_geometry(app, monkeypatch):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    import app.job_queue as jq
    got = _capture(monkeypatch)
    with app.app_context():
        monkeypatch.setattr(jq.queue_manager, 'cancel_job', lambda *a, **k: True)
        ds = _ds_with_ref(svc)
        svc.generate_variations_zimage(
            LOCAL_USER, ds.id,
            [{'label': 'Body standing, front', 'framing': 'body',
              'prompt': 'full body shot'}], 1)
        img = FaceDatasetImage.query.filter_by(dataset_id=ds.id).first()
        got.clear()
        svc.regenerate_image(LOCAL_USER, img.id)
    assert got[0]['framing'] == 'body'
    assert got[0]['aspect'] and ':' in got[0]['aspect']
    assert 'watermark' in got[0]['negative_prompt']
