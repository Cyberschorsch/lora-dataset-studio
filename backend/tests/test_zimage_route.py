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
