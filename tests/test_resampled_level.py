import numpy as np
import pytest
import soundfile as sf
import soxr
from earshot.forge import bank, sfx
from earshot.forge.polish import lufs_max

@pytest.mark.parametrize('sr', bank.SRS)
@pytest.mark.parametrize('peak_db,weight', [(0, None), (-1, None), (-1, 0.5)])
def test_final_sample_peak_and_metadata(sr, peak_db, weight):
    y, meta = bank.render('click/tink', sr=sr, peak_db=peak_db, weight=weight, checks=False)
    target = bank.level(peak_db, weight)
    assert y.dtype == np.float32 and y.ndim == 2 and y.shape[1] == 2
    assert float(np.max(np.abs(y))) == pytest.approx(10 ** (target / 20), abs=1e-7)
    assert np.max(np.abs(y)) <= 1
    assert meta['peak_db'] == round(target, 2)
    assert all(meta[k] == v for k, v in bank.times(y, sr).items())
    analysis = y.astype(np.float64) if sr == bank.SR else soxr.resample(y.astype(np.float64), sr, bank.SR)
    assert meta['lufs_m_max'] == round(float(lufs_max(analysis)), 2)
    assert meta['checks'] is None

@pytest.mark.parametrize('sr', [44100, 88200, 96000])
def test_official_pcm24_writer_preserves_the_take_within_quantization(tmp_path, sr):
    y, meta = bank.render('click/tink', sr=sr, peak_db=0)
    result = sfx.sfx_render('click/tink', out=str(tmp_path / 'take.wav'), sr=sr, peak_db=0)
    saved, read_sr = sf.read(result['file'], dtype='float32')
    assert read_sr == sr and sf.info(result['file']).subtype == 'PCM_24'
    np.testing.assert_allclose(saved, y, atol=2**-23, rtol=0)
    assert meta == result['meta']
    assert meta['checks'] == sfx.check(y, sr=sr, category='click', sound='click/tink')

@pytest.mark.parametrize('sound,options', [
    ('tap/finger', {'dur': 0.12}),
    ('whoosh_fast/crisp', {'dur': 0.4, 'apex': 0.2}),
    ('pluck/kalimba', {'pitch': 'E5', 'dur': 0.3}),
])
def test_other_shapes_and_knobs_keep_output_level(sound, options):
    native, _ = bank.render(sound, checks=False, **options)
    y, meta = bank.render(sound, sr=44100, peak_db=-1, checks=False, **options)
    assert y.ndim == native.ndim and y.shape[1:] == native.shape[1:]
    assert abs(len(y) - round(len(native) * 44100 / 48000)) <= 1
    assert np.max(np.abs(y)) == pytest.approx(10 ** (-1 / 20), abs=1e-7)
    assert all(meta[k] == v for k, v in bank.times(y, 44100).items())
    assert all(k in meta['edits'] for k in options)
