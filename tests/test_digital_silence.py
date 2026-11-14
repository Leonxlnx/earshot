"""Digital silence is simultaneous silence, not cancellation in a mono fold-down."""
import numpy as np
import pytest
import soundfile as sf
from earshot import features


def _signal(case):
    tone = np.tile(np.array([0.25, -0.25], dtype=np.float32), 12000)
    if case == 'mono':
        return tone, 0.0
    x = np.column_stack([tone, tone])
    expected = 0.0
    if case == 'antiphase':
        x[:, 1] *= -1
    elif case == 'one_channel_silent':
        x[:, 1] = 0
    elif case == 'alternating_channels':
        x[:12000, 0] = 0
        x[12000:, 1] = 0
    elif case == 'shared_padding':
        x[:6000] = 0
        x[6000:, 1] *= -1
        expected = 0.25
    elif case == 'all_silent':
        x[:] = 0
        expected = 1.0
    elif case == 'threshold':
        x[:12000] = 0.5e-6
        x[12000:, 0] = 1.5e-6
        x[12000:, 1] = 0
        expected = 0.5
    return x, expected


@pytest.mark.parametrize('case', ['mono', 'dual_mono', 'antiphase', 'one_channel_silent',
                                  'alternating_channels', 'shared_padding', 'all_silent', 'threshold'])
@pytest.mark.parametrize('kind', ['array', 'wav'])
def test_digital_silence_counts_all_channels_at_each_sample(tmp_path, case, kind):
    x, expected = _signal(case)
    src = x
    if kind == 'wav':
        src = str(tmp_path / f'{case}.wav')
        sf.write(src, x, 48000, subtype='FLOAT')
    result = features.measure(src, light=True)
    assert result['digital_silence_frac'] == expected
    if case in ('antiphase', 'dual_mono'):
        assert result['peak_dbfs'] == -12.0
        assert result['rms_dbfs'] == -12.0


def test_profile_measurement_cache_does_not_reuse_old_silence_semantics(tmp_path):
    from earshot import audio, profile

    path = str(tmp_path / 'antiphase.wav')
    sf.write(path, _signal('antiphase')[0], 48000, subtype='FLOAT')
    old_cache = profile._cache_file('audio', audio._key(path, audio.SR, 2), 'a3') + '.npz'
    assert profile.cache_paths(path)['audio'] != old_cache
