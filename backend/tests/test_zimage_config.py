from app import config as cfg


def test_zimage_in_default_engine_catalog():
    assert 'zimage' in cfg.DEFAULTS['engines']['enabled']


def test_zimage_settings_defaults():
    assert cfg.DEFAULTS['zimage']['denoise'] == 0.75
    assert cfg.DEFAULTS['zimage']['steps'] == 8


def test_new_engine_reaches_existing_install():
    # a user whose saved 'known' ledger predates zimage still gets it merged in
    conf = {'engines': {'enabled': ['chatgpt'], 'known': ['nanobanana', 'chatgpt', 'klein', 'krea']}}
    user = {'engines': {'enabled': ['chatgpt'], 'known': ['nanobanana', 'chatgpt', 'klein', 'krea']}}
    merged = cfg._merge_new_engines(conf, user)
    assert 'zimage' in merged['engines']['enabled']
