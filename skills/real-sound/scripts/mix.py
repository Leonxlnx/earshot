#!/usr/bin/env python
"""real-sound mixer: real recordings placed on the picture's cues. Nothing is synthesized here.

  python -I <skill>/scripts/mix.py [--root .] [--design audio/sounddesign.json] [--cues out/cues.json]
                                   [--out out/audio/soundtrack.wav] [--stems out/audio/stems] [--no-master]
                                   [--only keyboard] [--dump out/audio-review/dump] [--trace out/qa/adds.json]

Run it from the project root (or pass --root). Every relative path (design, cues, outputs, the "file" of every sample)
is relative to the root. Full reference with examples: references/design-format.md.

The cue file (JSON, from the picture's timeline): {fps, total (frames), events: [{f, kind, id?, weight?, pan?, ...}]}
(or a list whose first element is that object). A design without rules needs no cue file.

The design file (JSON) describes, in film seconds:
  duration  film length in seconds (default: cue total / fps)
  cues      optional default cue file path (the --cues flag wins)
  include   [other design fragments to merge: samples / slices / pools / rules / place / ambience / music]
  samples   name -> {file, start?, end?, gain_db?, norm_peak_db?, mono?, fade_in?, fade_out?, eq?, pitch?,
                     glide?: [from, to] semitones (tape-style varispeed across the sample), env?: [[t, dB], ...]}
            a trimmed piece of a downloaded recording (file paths are relative to the project root)
  slices    name -> {file, start?, end?, pre?, len?, min_gap?, max?, pick?, ...sample opts}
            cut a recording of many hits (typing, clicks) into single hits at its detected onsets: name#0, name#1...
  pools     name -> [sample names; "slicegroup#*" expands to all of its slices]
  rules     [{match: {kind?, id?, from?, to?}, pool, mode: cycle|hash|fixed, gain_db?, weight_db?, jitter_db?,
              pitch?: [lo, hi] semitones, pitch_ramp?: [a, b], gain_ramp?: [a, b] (across the matched events),
              pan?: "cue"|number, pan_scale?, align: attack|peak|end|start, offset?, bus?}]
            every cue event matching a rule gets a sound from the rule's pool (several rules may layer one event)
  place     [{sample, at, gain_db?, pan?, align?, offset?, pitch?, bus?}]   sounds with no cue event
  ambience  [{name?, file, src_start?, src_end?, at, until, fade_in?, fade_out?, gain_db?, eq?, width?, auto?: [[t, db], ...]}]
            (src_start..src_end is the clean window; it is looped with a 1.5 s crossfade if the bed is longer)
  music     {file, src_bpm, target_bpm?, stretch?: "rubberband"|"none", gain_db?, eq?,
             segments: [{src_from, src_to, at, fade_in?, fade_out?, gain_db?}]}   src_* in ORIGINAL source seconds
  buses     {sfx|amb|music: {gain_db?, eq?, auto?: [[t, db], ...]}}
  master    {lufs?, ceiling?, no_glue?, no_comp?, mono_hz?}
Processing is limited to trimming, fades, gain, pan, EQ, resampling (pitch variation), tempo stretch of the music,
summing and mastering (master.py next to this script). Writes the soundtrack, stems (sfx.wav, music.wav, amb.wav),
out/qa/mix_report.json (every placement, unsounded cue events, bus levels, files_used for the credits) and
out/qa/cues_sounded.json (the cue events that received a sound, for an A/V sync audit).

--only NAME   solo render: keep only the include fragments whose path contains NAME (no master; the unmastered mix is
              written to --out, default out/audio-review/solo/<NAME>.wav). Variation is keyed on rule names, so a solo
              render plays exactly the same sounds as the full mix.
--no-master   write the unmastered mix to --out (with master: --out is mastered, the unmastered sum is premaster.wav
              next to it).
--dump DIR    write every sample / slice of the design as its own WAV + _samples.json (check trims), then exit.
--trace FILE  also write every Mix.add() (bus, start sample, length, gain, pan, sample, rule) for QA attribution.
"""
import argparse, hashlib, json, math, os, re, subprocess, sys, zlib
import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000
ROOT = os.getcwd()  # the project root (set by --root)
CACHE = os.path.join(ROOT, 'out', 'audio', 'cache')
MASTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'master.py')


def P(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def db2a(d):
    return 10.0 ** (d / 20.0)


# ------------------------------------------------------------------ decoding (cached)
_dec = {}


def decode(path):
    path = P(path)
    if path in _dec:
        return _dec[path]
    os.makedirs(CACHE, exist_ok=True)
    st = os.stat(path)
    key = hashlib.sha1(f'{path}|{st.st_size}|{st.st_mtime}'.encode()).hexdigest()[:16]
    npy = os.path.join(CACHE, key + '.npy')
    if os.path.exists(npy):
        x = np.load(npy)
    else:
        p = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-f', 'f32le', '-acodec', 'pcm_f32le',
                            '-ac', '2', '-ar', str(SR), '-'], capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f'decode failed {path}: {p.stderr.decode(errors="replace")[:300]}')
        x = np.frombuffer(p.stdout, np.float32).reshape(-1, 2).copy()
        tmp = f'{npy}.{os.getpid()}.tmp.npy'
        np.save(tmp, x)
        os.replace(tmp, npy)  # atomic: other mixers may read the cache at the same time
    _dec[path] = x
    return x


def stretched(path, ratio):
    """Whole-file tempo change with Rubber Band (ffmpeg filter). ratio = target_bpm / src_bpm (speed factor)."""
    if abs(ratio - 1) < 1e-6:
        return decode(path)
    src = P(path)
    st = os.stat(src)
    key = hashlib.sha1(f'{src}|{st.st_size}|{st.st_mtime}|{ratio:.6f}|rb2'.encode()).hexdigest()[:16]
    out = os.path.join(CACHE, f'stretch_{key}.wav')
    if not os.path.exists(out):
        os.makedirs(CACHE, exist_ok=True)
        af = (f'rubberband=tempo={ratio:.6f}:transients=mixed:detector=compound:phase=laminar:window=standard:'
              f'smoothing=off:formant=shifted:pitchq=quality:channels=together')
        tmp = out.replace('.wav', f'.{os.getpid()}.tmp.wav')
        r = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', src, '-af', af, '-ar', str(SR), '-ac', '2',
                            '-c:a', 'pcm_f32le', tmp], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError('rubberband failed: ' + r.stderr.decode(errors='replace')[:400])
        os.replace(tmp, out)
    return decode(out)


# ------------------------------------------------------------------ DSP helpers (processing only)
def rc_fade(n, rising=True):
    if n <= 0:
        return np.ones(0)
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    return w if rising else w[::-1]


def apply_fades(x, fin, fout):
    x = x.copy()
    a, b = int(fin * SR), int(fout * SR)
    a, b = min(a, len(x)), min(b, len(x))
    if a:
        x[:a] *= rc_fade(a)[:, None]
    if b:
        x[len(x) - b:] *= rc_fade(b, False)[:, None]
    return x


def biquad(kind, f0, q=0.707, gain_db=0.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    if kind == 'peak':
        b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
    elif kind == 'lowshelf':
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * c + s), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - s)]
        a = [(A + 1) + (A - 1) * c + s, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - s]
    elif kind == 'highshelf':
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * c + s), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - s)]
        a = [(A + 1) - (A - 1) * c + s, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - s]
    else:
        raise ValueError(kind)
    return np.array(b) / a[0], np.array(a) / a[0]


def eq(x, spec):
    if not spec:
        return x
    y = x.astype(np.float64)
    if spec.get('hp'):
        sos = signal.butter(spec.get('hp_order', 2), spec['hp'], 'highpass', fs=SR, output='sos')
        y = signal.sosfilt(sos, y, axis=0)
    if spec.get('lp'):
        sos = signal.butter(spec.get('lp_order', 2), spec['lp'], 'lowpass', fs=SR, output='sos')
        y = signal.sosfilt(sos, y, axis=0)
    for f0, q, g in spec.get('peq', []):
        b, a = biquad('peak', f0, q, g); y = signal.lfilter(b, a, y, axis=0)
    if spec.get('lowshelf'):
        f0, g = spec['lowshelf']; b, a = biquad('lowshelf', f0, 0.707, g); y = signal.lfilter(b, a, y, axis=0)
    if spec.get('highshelf'):
        f0, g = spec['highshelf']; b, a = biquad('highshelf', f0, 0.707, g); y = signal.lfilter(b, a, y, axis=0)
    return y.astype(np.float32)


def pan_gains(p):
    p = max(-1.0, min(1.0, float(p)))
    th = (p + 1) * np.pi / 4
    return np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)


def resample_pitch(x, semis):
    if abs(semis) < 1e-4:
        return x
    import soxr
    r = 2 ** (semis / 12.0)
    return soxr.resample(x, SR * r, SR, quality='VHQ').astype(np.float32)


def varispeed(x, s0, s1):
    """tape-style pitch glide from s0 to s1 semitones across the sample (resampling a recording; no synthesis)"""
    import soxr
    r0, r1 = 2 ** (s0 / 12.0), 2 ** (s1 / 12.0)
    up = 4
    xo = soxr.resample(x.astype(np.float64), SR, SR * up, quality='VHQ')
    rmax = max(r0, r1, 1.0)
    if rmax > 1.0:  # band-limit before reading faster, so the glide never aliases
        sos = signal.butter(8, 0.95 * (SR / 2) / rmax, 'lowpass', fs=SR * up, output='sos')
        xo = signal.sosfiltfilt(sos, xo, axis=0)
    L = max(2, int(len(x) / ((r0 + r1) / 2)))
    r = np.linspace(r0, r1, L)
    pos = np.concatenate([[0.0], np.cumsum(r[:-1])]) * up
    pos = np.clip(pos, 0, len(xo) - 1.001)
    i = pos.astype(np.int64)
    f = (pos - i)[:, None]
    y = xo[i] * (1 - f) + xo[np.minimum(i + 1, len(xo) - 1)] * f
    return y.astype(np.float32)


def env_index(x, mode):
    m = np.abs(x).max(axis=1)
    if mode == 'start':
        return 0
    if mode == 'end':
        nz = np.where(m > m.max() * 0.003)[0]
        return int(nz[-1]) if len(nz) else len(x) - 1
    w = int(0.001 * SR) if mode == 'attack' else int(0.005 * SR)
    n = len(m) // w
    if n == 0:
        return 0
    e = m[: n * w].reshape(n, w).max(axis=1)
    if mode == 'peak':
        return int(np.argmax(e) * w + w // 2)
    return int(np.argmax(e >= e.max() * 0.1) * w)  # attack: first 1 ms within 20 dB of the peak


def auto_curve(points, n, default=0.0):
    """piecewise-linear dB automation over film time -> linear gain per sample"""
    if not points:
        return None
    pts = sorted(points)
    t = np.arange(n) / SR
    d = np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
    return db2a(d).astype(np.float32)


# ------------------------------------------------------------------ samples
class Lib:
    def __init__(self, design):
        self.d = design
        self.samples = {}
        self.meta = {}
        for name, s in (design.get('samples') or {}).items():
            self.samples[name] = self._make(s)
            self.meta[name] = {'file': s['file'], 'start': s.get('start', 0), 'end': s.get('end'), 'pitch': s.get('pitch', 0)}
        for name, s in (design.get('slices') or {}).items():
            for i, (x, t0) in enumerate(self._slice(s)):
                self.samples[f'{name}#{i}'] = x
                self.meta[f'{name}#{i}'] = {'file': s['file'], 'start': round(t0, 4), 'pitch': s.get('pitch', 0)}
        self.pools = {}
        for name, items in (design.get('pools') or {}).items():
            out = []
            for it in items:
                if it.endswith('#*'):
                    g = it[:-2]
                    out += sorted([k for k in self.samples if k.startswith(g + '#')], key=lambda k: int(k.split('#')[1]))
                else:
                    if it not in self.samples:
                        raise KeyError(f'pool {name}: unknown sample {it}')
                    out.append(it)
            if not out:
                raise ValueError(f'pool {name} is empty')
            self.pools[name] = out
        self._pitched = {}

    def _finish(self, x, s):
        if s.get('mono'):
            m = x.mean(axis=1, keepdims=True)
            x = np.repeat(m, 2, axis=1)
        if s.get('reverse'):
            raise ValueError('reverse is not used')
        x = eq(x, s.get('eq'))
        if s.get('pitch'):
            x = resample_pitch(x, s['pitch'])
        if s.get('glide'):  # [from, to] semitones across the sample
            x = varispeed(x, *s['glide'])
        if s.get('env'):  # [[t_seconds_from_sample_start, dB], ...] gain envelope inside the sample
            pts = sorted(s['env'])
            t = np.arange(len(x)) / SR
            x = x * db2a(np.interp(t, [p[0] for p in pts], [p[1] for p in pts]))[:, None].astype(np.float32)
        if s.get('norm_peak_db') is not None:
            pk = float(np.abs(x).max()) or 1.0
            x = x * (db2a(s['norm_peak_db']) / pk)
        x = apply_fades(x, s.get('fade_in', 0.002), s.get('fade_out', 0.012))
        return (x * db2a(s.get('gain_db', 0.0))).astype(np.float32)

    def _make(self, s):
        x = decode(s['file'])
        a = int(s.get('start', 0) * SR)
        b = int(s['end'] * SR) if s.get('end') is not None else len(x)
        return self._finish(x[a:b], s)

    def _slice(self, s):
        import librosa
        x = decode(s['file'])
        a = int(s.get('start', 0) * SR)
        b = int(s['end'] * SR) if s.get('end') is not None else len(x)
        seg = x[a:b]
        mono = seg.mean(axis=1)
        hop = 128
        on = librosa.onset.onset_detect(y=mono, sr=SR, hop_length=hop, backtrack=True, units='samples',
                                        delta=s.get('delta', 0.07), wait=int(s.get('min_gap', 0.06) * SR / hop))
        on = [int(o) for o in on]
        # loudness gate: drop onsets far below the loudest (handling noise, the far keys)
        pk = []
        for o in on:
            w = mono[o: o + int(0.03 * SR)]
            pk.append(float(np.abs(w).max()) if len(w) else 0)
        if pk:
            top = max(pk)
            on = [o for o, p in zip(on, pk) if p >= top * db2a(s.get('gate_db', -18))]
        pre, ln = int(s.get('pre', 0.004) * SR), int(s.get('len', 0.16) * SR)
        out = []
        for i, o in enumerate(on):
            st = max(0, o - pre)
            en = min(len(seg), o + ln)
            if s.get('stop_before_next', True) and i + 1 < len(on):
                en = min(en, on[i + 1] - pre)
            if en - st < int(0.02 * SR):
                continue
            out.append((self._finish(seg[st:en], {**s, 'fade_out': s.get('fade_out', 0.02)}), (a + st) / SR))
        if s.get('pick') is not None:
            out = [out[i] for i in s['pick'] if i < len(out)]
        if s.get('max'):
            out = out[: s['max']]
        if not out:
            raise ValueError(f'slices {s["file"]}: no onsets found')
        return out

    def get(self, name, semis=0.0):
        if abs(semis) < 1e-3:
            return self.samples[name]
        k = (name, round(semis, 2))
        if k not in self._pitched:
            self._pitched[k] = resample_pitch(self.samples[name], k[1])
        return self._pitched[k]


# ------------------------------------------------------------------ the mix
def h01(s):
    return (zlib.crc32(s.encode()) & 0xffffffff) / 0xffffffff


class Mix:
    def __init__(self, dur):
        self.n = int(round(dur * SR))
        self.bus = {k: np.zeros((self.n, 2), np.float32) for k in ('sfx', 'amb', 'music')}
        self.log = []

    def add(self, bus, x, start, gain=1.0, pan=0.0, **meta):
        self.log.append({'bus': bus, 'start': int(start), 'n': int(len(x)), 'gain': float(gain), 'pan': float(pan), **meta})
        if start >= self.n or start + len(x) <= 0:
            return
        a, b = max(0, start), min(self.n, start + len(x))
        seg = x[a - start: b - start] * gain
        if pan:
            gl, gr = pan_gains(pan)
            seg = seg * np.array([gl, gr], np.float32)
        self.bus[bus][a:b] += seg


def match(rule, e, fps):
    m = rule.get('match', {})
    if 'kind' in m and not re.fullmatch(m['kind'], e['kind']):
        return False
    if 'id' in m and not re.search(m['id'], e.get('id', '')):
        return False
    t = e['f'] / fps
    if 'from' in m and t < m['from'] - 1e-6:
        return False
    if 'to' in m and t >= m['to'] - 1e-6:
        return False
    if 'variant' in m and e.get('variant') != m['variant']:
        return False
    return True


def load_design(path, only=None):
    """the design file, with its "include" fragments merged in (samples / slices / pools / rules / place)"""
    if not os.path.exists(P(path)):
        sys.exit(f'design file not found: {P(path)}')
    d = json.load(open(P(path), encoding='utf-8'))
    for inc in d.get('include', []):
        if only and only not in inc:
            continue
        f = load_design(inc)
        for k in ('samples', 'slices', 'pools'):
            for name, v in (f.get(k) or {}).items():
                if name in d.setdefault(k, {}):
                    raise KeyError(f'{inc}: {k} "{name}" is defined twice')
                d[k][name] = v
        for k in ('rules', 'place', 'ambience'):
            d[k] = (f.get(k) or []) + (d.get(k) or [])
        if f.get('music') and not d.get('music'):
            d['music'] = f['music']
    return d


def missing_files(design):
    """every recording the design references that is not on disk, as 'kind name: path' lines"""
    refs = [(f'sample {n}', s.get('file')) for n, s in (design.get('samples') or {}).items()]
    refs += [(f'slices {n}', s.get('file')) for n, s in (design.get('slices') or {}).items()]
    refs += [(f"ambience {a.get('name') or i}", a.get('file')) for i, a in enumerate(design.get('ambience') or [])]
    if design.get('music'):
        refs.append(('music', design['music'].get('file')))
    return [f'{what}: {f}' for what, f in refs if not f or not os.path.isfile(P(f))]


def check_files(design):
    miss = missing_files(design)
    if miss:
        sys.exit(f'{len(miss)} recording(s) referenced by the design are missing (paths relative to --root {ROOT}):\n  '
                 + '\n  '.join(miss))


def dump(args):
    """write every sample / slice of the design as its own WAV + measurements, to check the trims"""
    design = load_design(args.design, args.only)
    check_files(design)
    lib = Lib(design)
    os.makedirs(P(args.dump), exist_ok=True)
    rows = []
    for name, x in lib.samples.items():
        fn = re.sub(r'[^A-Za-z0-9._#-]+', '_', name).replace('#', '-') + '.wav'
        sf.write(os.path.join(P(args.dump), fn), x, SR, subtype='PCM_24')
        m = np.abs(x).max(axis=1)
        pk = float(m.max()) or 1e-9
        a = env_index(x, 'attack')
        rows.append({'name': name, 'file': fn, 'src': lib.meta.get(name), 'dur': round(len(x) / SR, 3),
                     'peak_dbfs': round(20 * math.log10(pk), 1),
                     'rms_dbfs': round(10 * math.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-20), 1),
                     'attack_ms': round(a / SR * 1000, 1), 'peak_ms': round(env_index(x, 'peak') / SR * 1000, 1),
                     'head_db': round(20 * math.log10(float(m[: int(0.002 * SR)].max()) / pk + 1e-9), 1),
                     'tail_db': round(20 * math.log10(float(m[-int(0.02 * SR):].max()) / pk + 1e-9), 1)})
    for name, items in lib.pools.items():
        rows.append({'pool': name, 'size': len(items), 'items': items})
    out = os.path.join(P(args.dump), '_samples.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print(f'dumped {len(lib.samples)} samples, {len(lib.pools)} pools -> {out}')
    for r in rows:
        if 'pool' in r:
            print(f"  pool {r['pool']}: {r['size']}")
        else:
            print(f"  {r['name']:<34} {r['dur']:6.3f}s pk {r['peak_dbfs']:6.1f} rms {r['rms_dbfs']:6.1f} "
                  f"att {r['attack_ms']:6.1f}ms head {r['head_db']:6.1f} tail {r['tail_db']:6.1f}")


def run(args):
    if args.dump:
        return dump(args)
    design = load_design(args.design, args.only)
    cue_path = args.cues or design.get('cues') or 'out/cues.json'
    if os.path.exists(P(cue_path)):
        cues = json.load(open(P(cue_path)))
        cue = cues[0] if isinstance(cues, list) else cues
    elif design.get('rules'):
        sys.exit(f'cue file {cue_path} not found (the design has cue rules); pass --cues')
    else:
        cue = {'fps': 60, 'events': []}
    fps = cue.get('fps', 60)
    if 'duration' not in design and 'total' not in cue:
        sys.exit('set "duration" (seconds) in the design or "total" (frames) in the cue file')
    dur = design.get('duration', cue.get('total', 0) / fps)
    check_files(design)
    lib = Lib(design)
    mix = Mix(dur)
    placed_events = []
    report = {'placements': [], 'unsounded_events': [], 'buses': {}}

    # --- cue-driven effects
    events = sorted(cue['events'], key=lambda e: e['f'])
    counters, last = {}, {}
    sounded = set()
    for ri, rule in enumerate(design.get('rules', [])):
        pool = lib.pools[rule['pool']]
        matched = [e for e in events if match(rule, e, fps)]
        rk = rule.get('name', ri)  # per-event variation keyed on the rule's name: identical solo and merged
        for mi, e in enumerate(matched):
            mode = rule.get('mode', 'cycle')
            if mode == 'fixed':
                name = pool[0]
            elif mode == 'hash':
                name = pool[int(h01(f'{e.get("id")}|{e["f"]}|{rk}') * len(pool)) % len(pool)]
            else:
                c = counters.get(ri, int(rule.get('cycle_start', 0)))
                name = pool[c % len(pool)]
                if len(pool) > 1 and name == last.get(ri):
                    c += 1; name = pool[c % len(pool)]
                counters[ri] = c + 1
                last[ri] = name
            lo, hi = rule.get('pitch', [0, 0])
            semis = lo + (hi - lo) * h01(f'p|{e.get("id")}|{e["f"]}|{rk}') if hi != lo else lo
            if rule.get('pitch_ramp'):  # climbs across the matched events in time order (growth, stacking)
                r0, r1 = rule['pitch_ramp']
                semis += r0 + (r1 - r0) * (mi / max(1, len(matched) - 1))
            if rule.get('gain_ramp'):
                g0, g1 = rule['gain_ramp']
                rule_gain_ramp = g0 + (g1 - g0) * (mi / max(1, len(matched) - 1))
            else:
                rule_gain_ramp = 0.0
            g = rule.get('gain_db', 0.0) + rule_gain_ramp
            if rule.get('weight_db'):
                g += rule['weight_db'] * math.log2(max(e.get('weight', 1.0), 1e-3) / rule.get('weight_ref', 1.0))
            if rule.get('jitter_db'):
                g += (h01(f'g|{e.get("id")}|{e["f"]}|{rk}') * 2 - 1) * rule['jitter_db']
            pan = rule.get('pan', 'cue')
            pan = (e.get('pan', 0.0) * rule.get('pan_scale', 1.0)) if pan == 'cue' else float(pan)
            x = lib.get(name, semis)
            align = rule.get('align', 'attack')
            t = e['f'] / fps + rule.get('offset', 0.0)
            start = int(round(t * SR)) - env_index(x, align)
            mix.add(rule.get('bus', 'sfx'), x, start, db2a(g), pan, sample=name, rule=rule.get('name', ri), kind=e['kind'])
            report['placements'].append({'t': round(t, 4), 'event': e.get('id'), 'kind': e['kind'], 'rule': rule.get('name', ri),
                                         'sample': name, 'gain_db': round(g, 2), 'pitch': round(semis, 2),
                                         'pitch_total': round(semis + (lib.meta.get(name, {}).get('pitch') or 0), 2),
                                         'pan': round(pan, 3), 'align': align})
            if rule.get('bus', 'sfx') == 'sfx' and not rule.get('no_cue'):
                key = (e['f'], e['kind'], e.get('id'))
                if key not in sounded:
                    sounded.add(key)
                    placed_events.append(e)
    for e in events:
        if (e['f'], e['kind'], e.get('id')) not in sounded:
            report['unsounded_events'].append({'t': round(e['f'] / fps, 3), 'kind': e['kind'], 'id': e.get('id')})

    # --- free placements
    for p in design.get('place', []):
        x = lib.get(p['sample'], p.get('pitch', 0.0))
        t = p['at'] + p.get('offset', 0.0)
        start = int(round(t * SR)) - env_index(x, p.get('align', 'attack'))
        mix.add(p.get('bus', 'sfx'), x, start, db2a(p.get('gain_db', 0.0)), p.get('pan', 0.0), sample=p['sample'], kind='place')
        report['placements'].append({'t': round(t, 4), 'event': None, 'kind': 'place', 'sample': p['sample'],
                                     'gain_db': p.get('gain_db', 0.0), 'bus': p.get('bus', 'sfx')})

    # --- ambience beds
    for a in design.get('ambience', []):
        x = decode(a['file'])
        s0 = int(a.get('src_start', 0) * SR)
        need = int((a['until'] - a['at']) * SR)
        src = x[s0: int(a['src_end'] * SR) if a.get('src_end') is not None else len(x)]  # the clean window (also the loop)
        looped = len(src) < need
        if looped:  # loop with a 1.5 s equal-power crossfade
            xf = int(1.5 * SR)
            out = src.copy()
            while len(out) < need:
                nxt = src.copy()
                fo = np.sqrt(rc_fade(xf, False))[:, None]; fi = np.sqrt(rc_fade(xf))[:, None]
                out[-xf:] = out[-xf:] * fo + nxt[:xf] * fi
                out = np.concatenate([out, nxt[xf:]])
            src = out
        seg = src[:need]
        if a.get('width') is not None:
            m, s = (seg[:, 0] + seg[:, 1]) / 2, (seg[:, 0] - seg[:, 1]) / 2 * a['width']
            seg = np.stack([m + s, m - s], axis=1)
        seg = eq(seg, a.get('eq'))
        seg = apply_fades(seg, a.get('fade_in', 1.0), a.get('fade_out', 1.0))
        start = int(round(a['at'] * SR))
        if a.get('auto'):
            curve = auto_curve([[t - a['at'], d] for t, d in a['auto']], len(seg))
            seg = seg * curve[:, None]
        mix.add('amb', seg.astype(np.float32), start, db2a(a.get('gain_db', 0.0)), name=a.get('name') or a['file'], kind='ambience')
        body = seg[int(min(len(seg) - 1, a.get('fade_in', 1.0) * SR)): max(1, len(seg) - int(a.get('fade_out', 1.0) * SR))]
        report.setdefault('ambience', []).append({
            'name': a.get('name'), 'file': a['file'], 'at': a['at'], 'until': a['until'], 'looped': looped,
            'body_rms_dbfs': round(10 * math.log10(float(np.mean(body.astype(np.float64) ** 2)) + 1e-20)
                                   + a.get('gain_db', 0.0), 1) if len(body) else None})

    # --- music
    mu = design.get('music')
    if mu:
        ratio = mu.get('target_bpm', 120) / mu['src_bpm'] if mu.get('stretch', 'rubberband') != 'none' else 1.0
        x = stretched(mu['file'], ratio)
        k = 1.0 / ratio  # source seconds -> stretched seconds
        for sg in mu['segments']:
            fin, fout = sg.get('fade_in', 0.008), sg.get('fade_out', 0.06)
            a = int(round(sg['src_from'] * k * SR)) - int(fin * SR)
            b = int(round(sg['src_to'] * k * SR)) + int(fout * SR)
            pre = int(fin * SR)
            if a < 0:
                pre += a; a = 0
            seg = apply_fades(x[a:b], fin if sg.get('fade_in_at_start', True) else 0, fout)
            if sg.get('tail_fade'):  # a longer musical fade over the end of the segment
                n = int(sg['tail_fade'] * SR)
                seg[-n:] *= rc_fade(n, False)[:, None]
            start = int(round(sg['at'] * SR)) - pre
            mix.add('music', seg.astype(np.float32), start, db2a(sg.get('gain_db', 0.0)), label=sg.get('label'), kind='music')
            report.setdefault('music_segments', []).append({**sg, 'stretched_from': round(a / SR, 4), 'film_start': round(start / SR, 4)})
        if mu.get('eq'):
            mix.bus['music'] = eq(mix.bus['music'], mu['eq'])
        mix.bus['music'] *= db2a(mu.get('gain_db', 0.0))

    # --- buses
    for name, b in (design.get('buses') or {}).items():
        if name.startswith('_'):  # a comment key
            continue
        if name not in mix.bus:
            sys.exit(f'buses: unknown bus "{name}" (sfx, amb or music; comment keys start with _)')
        y = eq(mix.bus[name], b.get('eq'))
        c = auto_curve(b.get('auto'), mix.n)
        if c is not None:
            y = y * c[:, None]
        mix.bus[name] = (y * db2a(b.get('gain_db', 0.0))).astype(np.float32)

    # every recording that is audible in the film (for CREDITS.md)
    used = {lib.meta[p['sample']]['file'] for p in report['placements'] if p.get('sample') in lib.meta}
    used |= {a['file'] for a in design.get('ambience', [])}
    if mu:
        used.add(mu['file'])
    report['files_used'] = sorted(u.replace('\\', '/') for u in used)
    total = mix.bus['sfx'] + mix.bus['amb'] + mix.bus['music']
    os.makedirs(P(args.stems), exist_ok=True)
    for name, y in mix.bus.items():
        sf.write(os.path.join(P(args.stems), f'{name}.wav'), y, SR, subtype='FLOAT')
        pk = float(np.abs(y).max())
        report['buses'][name] = {'peak_dbfs': round(20 * math.log10(max(pk, 1e-9)), 2),
                                 'rms_dbfs': round(10 * math.log10(float(np.mean(y.astype(np.float64) ** 2)) + 1e-20), 2)}
    # mastered: the unmastered sum goes next to --out as premaster.wav; --no-master / --only: it IS the output
    pre = P(args.out) if args.no_master else os.path.join(os.path.dirname(P(args.out)), 'premaster.wav')
    sf.write(pre, total, SR, subtype='FLOAT')
    report['premaster_peak_dbfs'] = round(20 * math.log10(max(float(np.abs(total).max()), 1e-9)), 2)
    # cues that received an effect, for the A/V audit
    cp = {'fps': fps, 'total': cue.get('total', int(dur * fps)), 'events': [
        {k: v for k, v in e.items() if k in ('f', 'kind', 'weight', 'id', 'pan', 'dur', 'apexFrac')} for e in placed_events]}
    for e in cp['events']:
        if e['kind'] in ('riser', 'swell'):
            e['ends'] = True
    os.makedirs(os.path.dirname(P(args.cues_out)) or '.', exist_ok=True)
    json.dump(cp, open(P(args.cues_out), 'w'), indent=0)
    os.makedirs(os.path.dirname(P(args.report)), exist_ok=True)
    json.dump(report, open(P(args.report), 'w'), indent=1)
    if args.trace:
        os.makedirs(os.path.dirname(P(args.trace)) or '.', exist_ok=True)
        json.dump(mix.log, open(P(args.trace), 'w'), indent=0)
    print(f'premaster {pre}  peak {report["premaster_peak_dbfs"]} dBFS  placements {len(report["placements"])}  '
          f'unsounded {len(report["unsounded_events"])}')
    for k, v in report['buses'].items():
        print(f'  bus {k}: {v}')
    if not args.no_master and not np.any(total):
        print('the mix is digital silence (every bus is zero): nothing to master, no soundtrack written'
              + (f' ({P(args.out)} is from an earlier render)' if os.path.exists(P(args.out)) else '')
              + '. Check the trims (--dump) and the gains.')
        sys.exit(1)
    if not args.no_master:
        m = design.get('master', {})
        cmd = [sys.executable, '-I', MASTER, pre, P(args.out), '--lufs', str(m.get('lufs', -14)), '--ceiling',
               str(m.get('ceiling', 0.77)), '--json', os.path.join(os.path.dirname(P(args.report)), 'master.json')]
        if m.get('no_glue', True):
            cmd.append('--no-glue')
        if m.get('no_comp'):
            cmd.append('--no-comp')
        if m.get('mono_hz') is not None:
            cmd += ['--mono-hz', str(m['mono_hz'])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-1500:])
        if r.returncode != 0:
            print('master gate FAILED')
            sys.exit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--root', default='.', help='project root: every relative path is resolved against it')
    ap.add_argument('--design', default='audio/sounddesign.json')
    ap.add_argument('--cues', help='cue file (default: the design\'s "cues", else out/cues.json)')
    ap.add_argument('--out', help='default out/audio/soundtrack.wav (solo: out/audio-review/solo/<only>.wav)')
    ap.add_argument('--stems', help='default out/audio/stems (solo: next to --out)')
    ap.add_argument('--no-master', action='store_true')
    ap.add_argument('--only', help='solo render of the include fragments whose path contains this text (no master)')
    ap.add_argument('--dump', help='write every sample of the design to this folder as WAV + _samples.json, then exit')
    ap.add_argument('--report', help='default out/qa/mix_report.json (solo: next to --out)')
    ap.add_argument('--cues-out', help='the cue events that received a sound (A/V audit); default out/qa/cues_sounded.json')
    ap.add_argument('--trace', help='write every Mix.add() record here (QA: click / masking attribution)')
    ap.add_argument('--cache', help='decode cache folder (default <root>/out/audio/cache)')
    ap.add_argument('--master-script', help='mastering script (default: master.py next to this file)')
    a = ap.parse_args()
    ROOT = os.path.abspath(a.root)
    CACHE = os.path.abspath(os.path.join(ROOT, a.cache)) if a.cache else os.path.join(ROOT, 'out', 'audio', 'cache')
    if a.master_script:
        MASTER = os.path.abspath(a.master_script)
    if a.only:
        a.no_master = True
        tag = re.sub(r'[^A-Za-z0-9._-]+', '_', a.only)
        a.out = a.out or f'out/audio-review/solo/{tag}.wav'
        base = os.path.splitext(a.out)[0]
        a.stems = a.stems or base + '_stems'
        a.report = a.report or base + '_report.json'
        a.cues_out = a.cues_out or base + '_cues.json'
    a.out = a.out or 'out/audio/soundtrack.wav'
    a.stems = a.stems or 'out/audio/stems'
    a.report = a.report or 'out/qa/mix_report.json'
    a.cues_out = a.cues_out or 'out/qa/cues_sounded.json'
    os.makedirs(os.path.dirname(P(a.out)) or '.', exist_ok=True)
    run(a)
