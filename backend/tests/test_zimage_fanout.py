from app.services import face_dataset_service as svc


def test_zimage_is_local_engine():
    assert 'zimage' in svc.LOCAL_ENGINES
    assert svc.ZIMAGE_ENGINE == 'zimage'


def test_image_engine_maps_zimage_tag():
    class _Img:
        klein_model = 'zimage'
    assert svc._image_engine(_Img()) == 'zimage'
