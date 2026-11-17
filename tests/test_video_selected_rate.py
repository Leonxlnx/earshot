"""Selected-track timing uses the decoder's rate. Synthetic media only; no models."""
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from earshot import audio, config, video


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class SelectedVideoRateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = tuple(getattr(config, k) for k in ('HOME', 'HF_CACHE', 'EMB_CACHE', 'DEC_CACHE'))
        for k, sub in zip(('HOME', 'HF_CACHE', 'EMB_CACHE', 'DEC_CACHE'), ('cache', 'cache/hf', 'cache/emb', 'cache/decoded')):
            setattr(config, k, str(self.root / sub))

    def tearDown(self):
        for k, value in zip(('HOME', 'HF_CACHE', 'EMB_CACHE', 'DEC_CACHE'), self.old):
            setattr(config, k, value)
        self.tmp.cleanup()

    def make(self, first=10, second=25, default='second', has_audio=False):
        path = str(self.root / f'{first}-{second}-{default}-{has_audio}.mkv')
        args = ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                f'color=black:s=32x32:r={first}:d=3', '-f', 'lavfi', '-i',
                f'color=black:s=64x64:r={second}:d=3']
        if has_audio:
            args += ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo:d=3']
        args += ['-map', '0:v', '-map', '1:v']
        if has_audio:
            args += ['-map', '2:a', '-c:a', 'pcm_s16le']
        for i in (0, 1):
            args += [f'-filter:v:{i}', "drawbox=color=white:t=fill:enable='between(t,1,1.99)'"]
            args += [f'-disposition:v:{i}', 'default' if default == ('first', 'second')[i] else '0']
        subprocess.run(args + ['-c:v', 'ffv1', path], check=True, capture_output=True)
        return path

    def test_multitrack_order_defaults_and_audio(self):
        for first, second in ((10, 25), (25, 10)):
            for default in ('first', 'second', 'none'):
                for has_audio in (False, True):
                    with self.subTest(first=first, second=second, default=default, has_audio=has_audio):
                        path = self.make(first, second, default, has_audio)
                        expected = first if default == 'first' else second
                        fps, d = video.frame_diff(path, cache=False)
                        self.assertEqual(fps, expected)
                        self.assertEqual(len(d), 3 * expected)
                        self.assertEqual(video.cuts(path, cache=False), [1, 2])

    def test_decoded_pixels_unchanged(self):
        path = self.make()
        original = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-an', '-vf',
                                   'scale=96:54:flags=area,format=gray', '-f', 'rawvideo', '-'],
                                  check=True, capture_output=True).stdout
        fps, d = video.frame_diff(path, cache=False)
        self.assertEqual(fps, 25)
        np.testing.assert_array_equal(d, video._frame_differences(original))

    def test_legacy_metadata_cache_is_not_reused(self):
        path = self.make()
        folder = self.root / 'cache/emb/video'
        folder.mkdir(parents=True)
        key = hashlib.sha1(f'{audio.file_id(path)}|96x54|v1'.encode()).hexdigest()[:20]
        np.savez(folder / f'diff_{key}.npz', fps=10, d=np.zeros(30))
        self.assertEqual(video.cuts(path), [1, 2])
        self.assertEqual(len(list(folder.glob('diff_*.npz'))), 2)

    def test_cache_hit_and_bypass(self):
        path = self.make()
        fps, d = video.frame_diff(path)
        with patch.object(video, '_decode_picture', side_effect=AssertionError('cache missed')):
            got_fps, got_d = video.frame_diff(path)
        self.assertEqual(got_fps, fps)
        np.testing.assert_array_equal(got_d, d)
        cached = next((self.root / 'cache/emb/video').glob('diff_*.npz'))
        np.savez(cached, fps=fps, d=np.zeros_like(d))
        self.assertEqual(video.cuts(path), [])
        self.assertEqual(video.cuts(path, cache=False), [1, 2])
        cached.write_bytes(b'truncated')
        self.assertEqual(video.cuts(path), [1, 2])

    def test_failed_decode_has_no_result_cache(self):
        path = self.root / 'bad.mkv'
        path.write_bytes(b'not media')
        with self.assertRaises(audio.AudioError):
            video.frame_diff(str(path))
        self.assertFalse(list((self.root / 'cache/emb/video').glob('diff_*.npz')))

    def test_missing_rate_has_no_result_cache(self):
        path = self.make()
        reply = (bytes(96 * 54), video._PictureLog(), 0)
        with patch.object(video, '_decode_picture', return_value=reply):
            with self.assertRaisesRegex(audio.AudioError, 'frame rate'):
                video.frame_diff(path)
        self.assertFalse(list((self.root / 'cache/emb/video').glob('diff_*.npz')))


class RateParsingTests(unittest.TestCase):
    @staticmethod
    def line(rate, name='showinfo@earshot_rate'):
        return f'[{name} @ 0x123] config in time_base: 1/1000, frame_rate: {rate}\n'.encode()

    def test_exact_rational_and_equivalent_repeated_config(self):
        self.assertEqual(video._decoded_fps(self.line('30000/1001'), 'x'), 30000 / 1001)
        self.assertEqual(video._decoded_fps(self.line('25/1') + self.line('50/2'), 'x'), 25)

    def test_unusable_or_changing_metadata_rejected(self):
        for raw in (b'', self.line('25/1', 'other'), self.line('0/1'), self.line('1/0'),
                    self.line('0/0'), self.line('1/' + '9' * 400), self.line('9' * 400 + '/1'),
                    self.line('25/1') + self.line('30/1')):
            with self.subTest(raw=raw):
                with self.assertRaises(audio.AudioError):
                    video._decoded_fps(raw, 'fixture.mkv')

    def test_valid_then_malformed_config_is_rejected(self):
        for bad in ('-1/1', 'N/A', 'nan', '25/1 garbage', '0/0', '0/1'):
            with self.subTest(bad=bad):
                with self.assertRaises(audio.AudioError):
                    video._decoded_fps(self.line('25/1') + self.line(bad), 'fixture.mkv')



class BoundedLogTests(unittest.TestCase):
    def test_chunked_configs_and_diagnostics_are_bounded(self):
        line = RateParsingTests.line('30000/1001')
        for size in (1, 3, 29, 8192):
            log = video._PictureLog()
            data = b'metadata: ' + b'x' * 20000 + b'\n' + line + b'n: ordinary frame\n' * 10000
            for start in range(0, len(data), size):
                log.feed(data[start:start + size])
                self.assertLessEqual(len(log.pending), log.LIMIT)
                self.assertLessEqual(len(log.tail), log.LIMIT)
            log.finish()
            self.assertEqual(log.rate('x'), 30000 / 1001)
            self.assertEqual(log.pending, b'')

    def test_bad_or_changing_config_remains_invalid_after_good_line(self):
        for bad in ('N/A', '-1/1', '30/1', '0/0', '25/1 garbage'):
            log = video._PictureLog()
            log.feed(RateParsingTests.line('25/1') + RateParsingTests.line(bad) + RateParsingTests.line('25/1'))
            log.finish()
            with self.assertRaises(audio.AudioError):
                log.rate('x')

    def test_long_config_rejected_and_unterminated_config_handled(self):
        log = video._PictureLog()
        line = RateParsingTests.line('25/1').rstrip(b'\n')
        log.feed(line)
        log.finish()
        self.assertEqual(log.rate('x'), 25)
        log = video._PictureLog()
        log.feed(line + b'x' * 10000 + b'\n')
        log.feed(RateParsingTests.line('25/1'))
        log.finish()
        with self.assertRaises(audio.AudioError):
            log.rate('x')


if __name__ == '__main__':
    unittest.main()
