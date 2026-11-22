"""Audio ranges follow decoded duration, not a longer container's duration."""
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from earshot import audio, config
from earshot.timeline import timeline, _load_range


class DecodedRangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = (config.HOME, config.HF_CACHE, config.EMB_CACHE, config.DEC_CACHE)
        config.HOME, config.HF_CACHE, config.EMB_CACHE, config.DEC_CACHE = (
            str(self.root / s) for s in ('cache', 'cache/hf', 'cache/emb', 'cache/decoded'))
        self.wav = str(self.root / 'audio.wav')
        with wave.open(self.wav, 'wb') as f:
            f.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
            f.writeframes(bytes(48000 * 4))
        self.media = str(self.root / 'long-picture.mkv')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                        'color=c=black:s=32x32:r=10:d=3', '-i', self.wav,
                        '-c:v', 'ffv1', '-c:a', 'copy', self.media], check=True, capture_output=True)
        self.assertEqual(audio.probe(self.media)['duration'], 3)

    def tearDown(self):
        config.HOME, config.HF_CACHE, config.EMB_CACHE, config.DEC_CACHE = self.old
        self.tmp.cleanup()

    def test_past_audio_end_rejected_not_cached(self):
        for bounds in ({'start': 2, 'end': 3}, {'start': 1}, {'start': 1, 'end': 2}):
            with self.subTest(bounds=bounds):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, 'nothing to analyse'):
                        timeline(self.media, detail='low', **bounds)
        self.assertFalse(list(Path(config.EMB_CACHE).rglob('*.json')))

    def test_actual_short_window_rejected(self):
        for bounds in ({'start': .95, 'end': 2}, {'start': .95}, {'start': .95, 'end': 1}):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, 'nothing to analyse'):
                    timeline(self.media, cache=False, detail='low', **bounds)

    def test_valid_window_matches_full_duration_and_cache(self):
        full = timeline(self.media, cache=False, detail='low')
        for bounds, expected in (({'start': 0, 'end': 3}, (0, 1)),
                                 ({'start': .5, 'end': 3}, (.5, 1)),
                                 ({'start': -.5, 'end': .5}, (0, .5)),
                                 ({'start': .5}, (.5, 1))):
            with self.subTest(bounds=bounds):
                first = timeline(self.media, detail='low', **bounds)
                second = timeline(self.media, detail='low', **bounds)
                self.assertEqual((first['start'], first['end']), expected)
                self.assertEqual(first['file_duration'], full['file_duration'])
                self.assertEqual(first['file_duration'], 1)
                self.assertEqual(first['duration'], expected[1] - expected[0])
                self.assertFalse(first['cached'])
                self.assertTrue(second['cached'])
                self.assertEqual(second['context'], first['context'])
                self.assertTrue(all(v == 0 for v in first['model_calls'].values()))

    def test_unknown_or_underreported_probe_duration_uses_samples(self):
        for duration in (0, .5, 3):
            with self.subTest(duration=duration):
                pr = dict(audio.probe(self.wav), duration=duration)
                x, c0, t0, t1, fdur = _load_range(self.wav, pr, .5, 2, reuse=False)
                self.assertEqual((len(x), c0, t0, t1, fdur), (48000, 0, .5, 1, 1))

    def test_context_slice_matches_public_audio_load(self):
        # Nonzero stereo data proves sample equality, not only zero-array shape.
        long_path = str(self.root / 'long.wav')
        n = 65 * audio.SR
        samples = np.zeros((n, 2), dtype='<i2')
        samples[:, 0] = np.arange(n) % 1000
        samples[:, 1] = -(np.arange(n) % 2000)
        with wave.open(long_path, 'wb') as f:
            f.setparams((2, 2, audio.SR, 0, 'NONE', 'not compressed'))
            f.writeframes(samples.tobytes())
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                x, c0, t0, t1, fdur = _load_range(long_path, audio.probe(long_path), 35, 36, reuse)
                self.assertEqual((c0, t0, t1, fdur), (5, 35, 36, 65))
                np.testing.assert_array_equal(x, audio.load(long_path, start=5, end=65, cache=reuse))

    def test_picture_only_short_window_unchanged(self):
        picture = str(self.root / 'picture.mkv')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', self.media,
                        '-an', '-c:v', 'copy', picture], check=True, capture_output=True)
        r = timeline(picture, start=1, end=1.05, cache=False, detail='low')
        self.assertEqual((r['start'], r['end'], r['duration']), (1, 1.05, .05))
        self.assertFalse(r['has_audio'])

    def test_invalid_plain_audio_ranges_still_rejected(self):
        for bounds in ({'start': 2}, {'start': .7, 'end': .5}, {'end': -1}, {'end': .05}):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, 'nothing to analyse'):
                    timeline(self.wav, cache=False, detail='low', **bounds)


if __name__ == '__main__':
    unittest.main(verbosity=2)
