"""earshot.sounds.atlas and the semantic blend of earshot.brief.rank: the atlas covers the catalogue with unit
vectors made by the model earshot loads; the blend keeps a named job's lexical winner, lets the semantic score decide
a request that only describes a sound, and never lets a design the hard filters exclude (chip sound, loop, a job
ruled out) win on how it sounds; `make` without listening loads no model and says when its pick is a guess.
Fast tests use the atlas file and stubbed semantic scores; `slow` ones load CLAP."""
import sys

import numpy as np
import pytest

from earshot import brief
from earshot.sounds import atlas


@pytest.fixture(scope='module')
def idx():
    from earshot.sounds import registry
    return brief.index(registry.rows())


@pytest.fixture(scope='module')
def at():
    return atlas.load()


def _fam(id_):
    return id_.split('/')[0]


# ---- the atlas file

def test_atlas_covers_the_catalogue(at):
    from earshot.sounds import registry
    missing = sorted(set(registry.ids()) - set(at['ids']))
    assert not missing, f'designs not in the atlas (rebuild it): {missing}'
    assert atlas.check(at) == []


def test_atlas_vectors_are_unit_and_distinct(at):
    v = at['vecs']
    assert v.shape == (len(at['ids']), 512) and np.isfinite(v).all()
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=5e-3)
    assert len(set(at['ids'])) == len(at['ids'])
    c = v @ v.T
    np.fill_diagonal(c, 0.0)
    assert c.max() < 0.9999                     # no two designs rendered to the same sound


def test_atlas_names_its_model(at):
    from earshot import config
    assert at['model'] == config.CLAP_MODEL and len(at['digest']) == 16


def test_stale_atlas_is_reported(at, monkeypatch):
    a = dict(at, digest='0' * 16, ids=at['ids'][1:], keys=at['keys'][1:])
    bad = atlas.check(a)
    assert bad and at['ids'][0] in bad[0]


def test_missing_atlas_is_a_clear_error(tmp_path):
    with pytest.raises(atlas.AtlasError, match='build'):
        atlas.load(str(tmp_path / 'none.npz'))


# ---- the blend, with a stubbed semantic score

def _z(idx, high=(), low=(), base=0.0):
    """{id: z}: every design `base`, the ids in `high` +4, in `low` -1.5"""
    z = {r['id']: base for r in idx[0]}
    z.update({i: 4.0 for i in high})
    z.update({i: -1.5 for i in low})
    return z


def test_rank_without_semantic_is_lexical(idx):
    b = brief.read('a soft success chime')
    assert brief.rank(b, idx=idx) == brief.rank(b, idx=idx, semantic=None)


@pytest.mark.parametrize('text', ['success chime', 'coin pickup', 'a deep impact for a logo reveal', 'rain on a window',
                                  'jump sound for a platformer'])
def test_named_job_keeps_its_lexical_winner(idx, text):
    b = brief.read(text)
    lex = brief.rank(b, idx=idx)[0]['id']
    other = next(r['id'] for r, t in zip(idx[0], idx[1]) if _fam(r['id']) != _fam(lex)
                 and not brief.excluded(b, r, t))
    got = brief.rank(b, idx=idx, semantic=_z(idx, high=[other], low=[lex]))[0]['id']
    assert _fam(got) == _fam(lex), (text, lex, got)


def test_described_sound_is_decided_by_the_atlas(idx):
    # no word of it names a design ("fading" and "air" are in fall/air's descriptions since the film rebuild)
    b = brief.read('a faint thin streak dissolving far away')
    for target in ('whoosh_fast/crisp', 'laser/beam', 'shimmer/m1'):
        ids = {r['id'] for r in idx[0]}
        if target not in ids:
            continue
        got = brief.rank(b, idx=idx, semantic=_z(idx, high=[target]))
        assert got[0]['id'] == target and got[0]['z'] == 4.0


def test_hard_filters_hold_against_the_semantic_score(idx):
    chip = next(r['id'] for r in idx[0] if brief.chip(r))
    plain = next(r['id'] for r in idx[0] if r['id'].startswith('coin/') and not brief.chip(r))
    got = brief.rank(brief.read('an 8-bit coin pickup'), idx=idx, semantic=_z(idx, high=[plain]))
    assert brief.chip(next(r for r in idx[0] if r['id'] == got[0]['id']))
    got = brief.rank(brief.read('a soft coin pickup'), idx=idx, semantic=_z(idx, high=[chip]))
    assert got[0]['id'] != chip
    shot = next(r['id'] for r in idx[0] if not r['loop'] and r['kind'] == 'hit')
    got = brief.rank(brief.read('a seamless loop of something calm'), idx=idx, semantic=_z(idx, high=[shot]))
    assert all(r['id'] != shot for r in got)
    err = next(r['id'] for r in idx[0] if r['id'].startswith('error/'))
    got = brief.rank(brief.read('a short alert, not an error'), idx=idx, semantic=_z(idx, high=[err]))
    assert got[0]['id'] != err


def test_blend_names_the_semantic_evidence(idx):
    b = brief.read('a plump warm double plink')
    got = brief.rank(b, idx=idx, top=None, semantic=_z(idx, high=['coin/bright']))
    top = next(r for r in got if r['id'] == 'coin/bright')
    assert any('sounds like it' in w for w in top['why']) and top['lex'] >= 0.0


# ---- make: no model without listening, and a guess says so

def test_make_without_listen_loads_no_model(tmp_path):
    from earshot.make import make
    r = make('a soft success chime', str(tmp_path / 's.wav'))
    assert r['files'] and not atlas.warm()
    assert 'torch' not in sys.modules


def test_guess_line():
    from earshot.make import guess
    b = {'listened': False}
    sure = [{'id': 'success/soft', 'score': 9.0, 'why': ['success']}, {'id': 'notify/soft', 'score': 4.0, 'why': []}]
    assert guess(b, sure) is None
    tied = [{'id': 'a/x', 'score': 5.0, 'why': ['success']}, {'id': 'b/y', 'score': 4.95, 'why': []},
            {'id': 'c/z', 'score': 1.0, 'why': []}]
    line = guess(b, tied)
    assert 'b/y' in line and 'c/z' in line and 'sound=' in line
    vague = [{'id': 'a/x', 'score': 2.0, 'why': ['airy']}, {'id': 'b/y', 'score': 1.0, 'why': []}]
    assert 'no job or sound word' in guess(b, vague)
    heard = [{'id': 'a/x', 'score': 2.0, 'why': ['airy'], 'z': 3.5}, {'id': 'b/y', 'score': 1.0, 'why': [], 'z': 0.0}]
    assert guess({'listened': True}, heard) is None


# ---- with the model

@pytest.mark.slow
@pytest.mark.parametrize('text,families', [
    ('steady rain falling on a window pane', ('rain',)),
    ('a deep low boom that drops in pitch', ('sub_drop', 'pro_sub_drop', 'impact', 'boom', 'cut_hit', 'explosion')),
    ('crackling campfire', ('fire',)),
    ('a fast airy whoosh passing by', ('whoosh_fast', 'whoosh', 'swish', 'whoosh_big', 'dash', 'swipe')),
])
def test_atlas_hears_descriptions(text, families):
    got = [_fam(r['id']) for r in atlas.nearest(text, top=3)]
    assert set(got) & set(families), (text, got)


@pytest.mark.slow
def test_semantic_ranking_is_fast_when_warm():
    import time
    atlas.nearest('a soft rounded pluck')
    t = time.perf_counter()
    atlas.zmap('a soft rounded pluck, a little brighter')
    assert time.perf_counter() - t < 2.0
