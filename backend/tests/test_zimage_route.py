"""Z-Image Turbo — generate-route dispatch + the missing-weights 409.

WHAT THIS IS FOR
-----------------
Z-Image Turbo is the THIRD local engine (after Klein and Krea 2 Edit), and the
only one whose route wiring is covered here: KNOWN_ENGINES/LOCAL_ENGINES
membership, the up-front preflight 409 (mirroring Klein's auto-download shape,
NOT Krea's manual-instructions shape — every Z-Image asset is a public direct
Hugging Face file setup_installer already knows how to fetch), and the batch
dispatch branch. Nothing here renders anything: not one GPU second.
"""
import os

import pytest


def test_zimage_is_a_known_local_engine():
    from app.services import face_dataset_service as svc
    assert 'zimage' in svc.KNOWN_ENGINES and 'zimage' in svc.LOCAL_ENGINES
    assert 'zimage' not in svc.API_ENGINES


def test_zimage_missing_response_autostarts_and_409s(app, monkeypatch):
    import app.routes.datasets as d
    from app.services.zimage_edit_helper import ZImageModelsMissing
    from app import capabilities
    monkeypatch.setattr(capabilities, 'resolve_comfyui_base',
                        lambda base: {'valid': True, 'resolved': '/x'})
    monkeypatch.setattr(d, '_autostart_zimage_downloads', lambda missing: (list(missing), False))
    with app.test_request_context():
        resp, status = d._zimage_missing_response(ZImageModelsMissing(['zimage_model']))
        body = resp.get_json()
    assert status == 409
    assert body['ok'] is False
    assert body['zimage_missing'] == ['zimage_model']
    assert 'zimage_model' in body['downloading']


def test_generating_on_zimage_without_weights_answers_one_409(client, monkeypatch):
    from app.services import zimage_edit_helper as zih
    import app.routes.datasets as d
    from app import capabilities
    monkeypatch.setattr(zih, 'zimage_missing_assets',
                        lambda: ['zimage_model', 'zimage_text_encoder', 'zimage_vae'])
    monkeypatch.setattr(d, '_autostart_zimage_downloads', lambda missing: (list(missing), False))
    monkeypatch.setattr(capabilities, 'resolve_comfyui_base',
                        lambda base: {'valid': True, 'resolved': '/x'})
    ds = client.post('/api/dataset/create',
                     json={'name': 'Z', 'trigger_word': 'ztrig'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds}/generate', json={
        'engine_batches': [{'generator': 'zimage',
                            'variations': [{'label': 'Bust, front', 'framing': 'bust',
                                            'prompt': 'upper body portrait'}]}],
        'multiplier': 1})
    assert r.status_code == 409
    assert r.get_json()['ok'] is False
    assert 'zimage_missing' in r.get_json()
    # nothing created — a preflight that leaves half a batch behind is worse than none
    assert client.get(f'/api/dataset/{ds}').get_json()['images'] == []


def test_regenerate_missing_weights_answers_409_not_500(app, client, monkeypatch):
    """Single-tile 🔁 regenerate must map ZImageModelsMissing to the same
    friendly 409 as the batch /generate route. Before this fix the regenerate
    route's preflight if/elif chain only knew about krea (explicit engine) and
    klein (node preflight) — an explicit engine='zimage' fell through to the
    klein branch (fail-open, no ComfyUI reachable) and on into
    svc.regenerate_image(), which raises ZImageModelsMissing once it tries to
    resolve the Z-Image assets; the except block didn't recognise that
    exception either, so it reached _map_error(e), which re-raises anything
    it doesn't know — a bare 500 instead of the actionable auto-download 409."""
    import io
    from PIL import Image
    from app.services import face_dataset_service as svc
    from app.services import zimage_edit_helper as zih
    from app.config import LOCAL_USER
    from app import capabilities
    monkeypatch.setattr(capabilities, 'resolve_comfyui_base',
                        lambda base: {'valid': True, 'resolved': '/x'})
    monkeypatch.setattr(zih, 'zimage_missing_assets', lambda: ['zimage_model'])
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ZRegen', 'zregentrig')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        buf = io.BytesIO()
        Image.new('RGB', (64, 64), (0, 128, 255)).save(buf, 'PNG')
        with open(os.path.join(d, 'ref.png'), 'wb') as fh:
            fh.write(buf.getvalue())
        ds.ref_filename = 'ref.png'
        img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                                   status='finished', variation_prompt='p')
        svc.db.session.add(img)
        svc.db.session.commit()
        img_id = img.id
    resp = client.post(f'/api/dataset/image/{img_id}/regenerate',
                       json={'engine': 'zimage'})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['ok'] is False
    assert 'zimage_model' in body['zimage_missing']
    with app.app_context():
        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert row.status == 'finished'   # blocked before any reset — untouched
