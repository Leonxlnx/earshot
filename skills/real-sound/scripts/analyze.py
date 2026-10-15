#!/usr/bin/env python
"""Objective analysis of downloaded audio assets (we cannot listen, so we measure).

  python -I <skill>/scripts/analyze.py FILE [FILE ...] [--music] [--png DIR] [--out OUT.json] [--json]

Prints one plain-language line per file (then a summary line; exit code 1 when a file is missing or does not decode);
--out writes every measurement as a JSON list, --json prints that list instead of the lines.
Every file is decoded with ffmpeg to 48 kHz float stereo (a mono source is duplicated at unity gain). Reports per file:
  format (codec, source rate, channels, bitrate), duration, peak / RMS / integrated LUFS, noise floor (10th percentile of
  20 ms RMS), clipping, DC, the first transient (attack) time, onset times, spectral centroid, bandwidth (95% rolloff),
  and "spikes" (sudden events far above the median level: useful to catch dogs, voices, bumps inside an ambience).
With --music also: tempo + beat times (librosa), tempo stability, key estimate (Krumhansl), loudness per 2 s and per
bar, and novelty-based section boundaries.
With --png: a spectrogram + waveform image per file (ffmpeg showspectrumpic), to look at with an image viewer.

Reading the numbers (rules of thumb for picking recordings):
  foley / SFX   noise_floor_dbfs < -55 between events, clipped_samples 0, rolloff95_hz > 12 kHz (not dull), a
                band_db_rel_1_4k['16-19k'] cliff far below '12-16k' means a low-bitrate encode; spikes_over_12db inside an
                ambience = single events (dog, voice, bump) to avoid or cut around.
  ambience      level_std_db_100ms small (steady bed), side_to_mid_db > -12 (real stereo; null = mono), no spikes,
                >= 40 s usable.
  music         tempo + ibi_std_ms small (steady grid), key, rms_db_per_bar (the arc), sections_at (edit points).
"""
import argparse, json, os, subprocess, sys, math, warnings
import numpy as np

SR = 48000


def probe(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                              'stream=codec_name,sample_rate,channels,bit_rate:format=bit_rate,duration',
                              '-of', 'json', path], capture_output=True, text=True, timeout=60).stdout
        j = json.loads(out)
        s = (j.get('streams') or [{}])[0]
        f = j.get('format') or {}
        brs = [int(b) for b in (s.get('bit_rate'), f.get('bit_rate')) if str(b).isdigit() and 0 < int(b) < 10 ** 7]
        return {'codec': s.get('codec_name'), 'src_rate': int(s.get('sample_rate', 0) or 0),
                'channels': int(s.get('channels', 0) or 0), 'kbps': round(brs[0] / 1000) if brs else None}
    except Exception as e:  # noqa
        return {'probe_error': str(e)}


def decode(path, channels=2):
    """(n, 2) float32 at SR; a mono source is duplicated at unity gain (ffmpeg's own upmix is 3 dB down)"""
    ch = 1 if channels == 1 else 2
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-map', '0:a:0', '-f', 'f32le', '-acodec', 'pcm_f32le',
                        '-ac', str(ch), '-ar', str(SR), '-'], capture_output=True, timeout=600)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(p.stderr.decode(errors='replace').strip()[:400] or 'no audio decoded')
    x = np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, ch)
    return np.repeat(x, 2, axis=1) if ch == 1 else x.copy()


def db(x):
    return float(20 * np.log10(max(float(x), 1e-12)))


def frame_rms(mono, win):
    n = len(mono) // win
    if n == 0:
        return np.array([np.sqrt(np.mean(mono ** 2) + 1e-20)])
    f = mono[: n * win].reshape(n, win)
    return np.sqrt(np.mean(f ** 2, axis=1) + 1e-20)


KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def key_estimate(chroma_mean):
    best = []
    for mode, prof in (('major', KK_MAJ), ('minor', KK_MIN)):
        for k in range(12):
            r = np.corrcoef(chroma_mean, np.roll(prof, k))[0, 1]
            best.append((float(r), f'{NAMES[k]} {mode}'))
    best.sort(reverse=True)
    return [{'key': b[1], 'r': round(b[0], 3)} for b in best[:3]]


def analyze(path, music=False):
    import librosa
    warnings.filterwarnings('ignore', message='n_fft=')  # very short sounds: librosa pads, the numbers stay usable
    info = {'file': path.replace('\\', '/'), **probe(path)}
    y = decode(path, info.get('channels'))
    mono = y.mean(axis=1)
    n = len(mono)
    info['duration'] = round(n / SR, 3)
    if n < SR * 0.02:
        info['error'] = 'too short'
        return info
    peak = float(np.max(np.abs(y)))
    info['peak_dbfs'] = round(db(peak), 2)
    info['rms_dbfs'] = round(db(np.sqrt(np.mean(y ** 2))), 2)
    info['clipped_samples'] = int(np.sum(np.abs(y) >= 0.999))
    info['dc'] = round(float(np.mean(mono)), 5)
    try:
        import pyloudnorm as pyln
        if n > SR * 0.45:
            v = float(pyln.Meter(SR).integrated_loudness(y.astype(np.float64)))
            info['lufs'] = round(v, 2) if math.isfinite(v) else None  # None: silence
    except Exception:
        pass
    r20 = frame_rms(mono, int(0.02 * SR))
    info['noise_floor_dbfs'] = round(db(np.percentile(r20, 10)), 1)
    info['loudest_20ms_dbfs'] = round(db(np.max(r20)), 1)
    info['dynamic_range_db'] = round(info['loudest_20ms_dbfs'] - info['noise_floor_dbfs'], 1)
    # stereo width: side/mid energy
    mid, side = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
    info['side_to_mid_db'] = (round(db(np.sqrt(np.mean(side ** 2))) - db(np.sqrt(np.mean(mid ** 2))), 1)
                              if np.any(side) else None)  # None: both channels identical (mono)
    # first transient: first 1 ms window whose envelope gets within 20 dB of the peak envelope
    r1 = frame_rms(mono, int(0.001 * SR))
    thr = np.max(r1) * 0.1
    idx = int(np.argmax(r1 >= thr))
    info['attack_at'] = round(idx * 0.001, 4)
    pk = int(np.argmax(r1))
    info['peak_at'] = round(pk * 0.001, 4)
    # spectral shape (on a 22.05k copy for speed)
    ys = librosa.resample(mono, orig_sr=SR, target_sr=22050) if n < SR * 400 else mono[:: 2]
    sr2 = 22050
    S = np.abs(librosa.stft(ys, n_fft=2048, hop_length=512))
    cen = librosa.feature.spectral_centroid(S=S, sr=sr2)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr2, roll_percent=0.95)[0]
    act = r20[: len(cen)] if len(r20) else None
    info['centroid_hz'] = int(np.median(cen))
    info['rolloff95_hz'] = int(np.percentile(roll, 90))
    # high-band levels relative to 1-4 kHz (an encoder lowpass shows up as a cliff, e.g. 16-19k far below 12-16k)
    m30 = mono[: min(n, SR * 60)]
    Sf = np.abs(np.fft.rfft(m30 * np.hanning(len(m30)))) ** 2
    fr = np.fft.rfftfreq(len(m30), 1 / SR)
    band = lambda a, b: float(np.mean(Sf[(fr >= a) & (fr < b)]) + 1e-30)
    ref = band(1000, 4000)
    info['band_db_rel_1_4k'] = {k: round(10 * np.log10(band(a, b) / ref), 1) for k, (a, b) in
                                {'40-120': (40, 120), '8-12k': (8000, 12000), '12-16k': (12000, 16000),
                                 '16-19k': (16000, 19000), '19-22k': (19000, 22000)}.items()}
    onsets = librosa.onset.onset_detect(y=ys, sr=sr2, units='time', backtrack=False)
    info['onset_count'] = int(len(onsets))
    info['onsets'] = [round(float(o), 3) for o in onsets[:40]]
    # spikes: 100 ms windows more than 12 dB above the median level
    r100 = frame_rms(mono, int(0.1 * SR))
    med = np.median(r100)
    sp = np.where(r100 > med * 4)[0]
    info['spikes_over_12db'] = [round(i * 0.1, 1) for i in sp[:30]]
    info['level_std_db_100ms'] = round(float(np.std(20 * np.log10(r100 + 1e-12))), 2)
    if music:
        tempo, beats = librosa.beat.beat_track(y=ys, sr=sr2, start_bpm=120, units='time', tightness=200)
        tempo = float(np.atleast_1d(tempo)[0])
        info['tempo'] = round(tempo, 2)
        if len(beats) > 4:
            ibi = np.diff(beats)
            info['tempo_from_ibi'] = round(60 / float(np.median(ibi)), 2)
            info['ibi_std_ms'] = round(float(np.std(ibi)) * 1000, 1)
        info['beats'] = [round(float(b), 3) for b in beats]
        chroma = librosa.feature.chroma_cqt(y=ys, sr=sr2, hop_length=512)
        info['key'] = key_estimate(chroma.mean(axis=1))
        r2 = frame_rms(mono, SR * 2)
        info['rms_db_per_2s'] = [round(db(v), 1) for v in r2]
        # loudness per bar (assuming 4/4 on the tracked beats)
        if len(beats) > 8:
            bars = beats[::4]
            pb = []
            for a, b in zip(bars[:-1], bars[1:]):
                seg = mono[int(a * SR): int(b * SR)]
                pb.append(round(db(np.sqrt(np.mean(seg ** 2) + 1e-20)), 1))
            info['bar_starts_if_beat0_is_downbeat'] = [round(float(b), 3) for b in bars]
            info['rms_db_per_bar'] = pb
        # sections: novelty on stacked chroma + mfcc
        try:
            mf = librosa.feature.mfcc(y=ys, sr=sr2, n_mfcc=13, hop_length=512)
            F = np.vstack([librosa.util.normalize(chroma, axis=1), librosa.util.normalize(mf, axis=1)])
            k = max(4, min(12, int(info['duration'] // 15)))
            bounds = librosa.segment.agglomerative(F, k)
            info['sections_at'] = [round(float(t), 2) for t in librosa.frames_to_time(bounds, sr=sr2, hop_length=512)]
        except Exception as e:
            info['sections_error'] = str(e)[:200]
    return info


def png(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(outdir, base + '_spec.png')
    p = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-lavfi',
                        'showspectrumpic=s=1400x420:mode=combined:color=intensity:scale=log:legend=1', out],
                       capture_output=True, timeout=300)
    if p.returncode or not os.path.exists(out):
        raise RuntimeError(p.stderr.decode(errors='replace').strip()[:200] or 'no image written')
    return out.replace('\\', '/')


def describe(r):
    """one plain-language line: format, level, noise, timing, tone, and what looks wrong"""
    if r.get('error'):
        return f"{os.path.basename(r['file'])}: ERROR {r['error']}"
    fmt = f"{r.get('codec')} {r.get('kbps') or '?'} kbps {(r.get('src_rate') or 0) / 1000:g} kHz {r.get('channels')}ch"
    lufs = f"{r['lufs']} LUFS" if r.get('lufs') is not None else 'LUFS n.a.'
    s = (f"{os.path.basename(r['file'])}: {fmt}, {r['duration']} s, peak {r['peak_dbfs']} dBFS, {lufs}, noise floor "
         f"{r['noise_floor_dbfs']} dBFS, attack {r['attack_at']} s, centroid {r['centroid_hz']} Hz, rolloff95 "
         f"{r['rolloff95_hz']} Hz, " + ('mono' if r.get('side_to_mid_db') is None else f"side/mid {r['side_to_mid_db']} dB"))
    if r.get('tempo'):
        s += f", {r.get('tempo_from_ibi', r['tempo'])} BPM (ibi std {r.get('ibi_std_ms')} ms), key {r['key'][0]['key']}"
    warn = []
    if r['peak_dbfs'] <= -90:
        warn.append('digital silence')
    if r.get('clipped_samples'):
        warn.append(f"{r['clipped_samples']} clipped samples")
    if r.get('kbps') and r.get('codec') == 'mp3' and r['kbps'] < 192:
        warn.append('low-bitrate mp3')
    if r.get('src_rate') and r['src_rate'] < 44100:
        warn.append(f"sample rate {r['src_rate']} Hz")
    if r['rolloff95_hz'] < 8000 and r['peak_dbfs'] > -90:
        warn.append('dull (little above 8 kHz)')
    if r.get('spikes_over_12db'):
        warn.append(f"{len(r['spikes_over_12db'])} spike(s) > 12 dB over the median level")
    return s + ('; CHECK: ' + ', '.join(warn) if warn else '')


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return None if isinstance(o, float) and not math.isfinite(o) else o


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('files', nargs='+')
    ap.add_argument('--music', action='store_true', help='also tempo, beats, key, loudness per bar, sections')
    ap.add_argument('--png', metavar='DIR', help='write a spectrogram image per file here')
    ap.add_argument('--out', metavar='OUT.json', help='write every measurement as a JSON list')
    ap.add_argument('--json', action='store_true', help='print the JSON list instead of one line per file')
    a = ap.parse_args()
    res = []
    for f in a.files:
        if not os.path.isfile(f):
            r = {'file': f, 'error': 'file not found'}
        else:
            try:
                r = analyze(f, a.music)
            except Exception as e:  # noqa: BLE001
                r = {'file': f, 'error': str(e)[:400]}
            if a.png:
                try:
                    r['png'] = png(f, a.png)
                except Exception as e:  # noqa: BLE001
                    r['png_error'] = str(e)[:200]
        r = clean(r)
        r['summary'] = describe(r)
        res.append(r)
        if not a.json:
            print(r['summary'].encode('ascii', 'replace').decode())
    bad = [os.path.basename(r['file']) for r in res if r.get('error')]
    summary = f"{len(res)} file(s) analysed, {len(bad)} failed" + (f" ({', '.join(bad[:6])})" if bad else '')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1)
        summary += f'; wrote {a.out}'
    print(json.dumps(res, indent=1, ensure_ascii=True) if a.json else summary)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
