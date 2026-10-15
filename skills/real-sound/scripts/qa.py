#!/usr/bin/env python
"""Objective QA of a real-sound mix: what the critic lenses measure, in one pass, tied to the placements that caused it.

  python -I <skill>/scripts/qa.py [--root .] [--design audio/sounddesign.json] [--report out/qa/mix_report.json]
                                  [--trace out/qa/adds.json] [--stems out/audio/stems] [--master out/audio/soundtrack.wav]
                                  [--cues out/cues.json] [--only NAME] [--cache out/audio/cache] [--video out/film.mp4]
                                  [--out out/qa/qa.json] [--layers] [--listen] [--realness] [--json]

Run it after `mix.py --trace out/qa/adds.json` (same design, same --only): the trace ties every click, masked hit and
harsh peak to the placement that caused it. With --only, the report / stems default to mix.py's solo outputs
(out/audio-review/solo/<only>_report.json, _stems) and there is no master (levels at stem scale). Without the trace
only clicks, loudness and coverage run. No models needed; --listen / --realness add earshot's ears when installed. Sections (status ok / warn / fail + a plain summary):
  clicks      > 14 kHz bursts concentrated in 1 ms (the av-audit detector) in the master and each stem, attributed to
              the placements whose start / end / body they sit in
  edges       used samples that start mid-sound or are cut mid-decay (unfaded head / tail level vs their peak, and the
              source just outside the trim), a second hit inside a sample, steps in a stem at a placement's start / end
              (second-difference ratio), placements cut by the film's ends
  audibility  every cue event's effect (its layers summed, rebuilt from the trace) vs the bed (music + amb) and vs
              everything else, in 1/3-octave bands over its loudest 60 ms and its first 20 ms: read >= 6 dB clear, 3-6 ok,
              0-3 barely, < 0 masked (margins capped at +-60 dB); an effect peaking under -70 dBFS is SILENT (a trim
              that missed the sound: fail); not judged when the render has no bed (a solo); --layers also lists single
              layers that add nothing measurable
  harshness   each event's 2-6 kHz peak (5 ms RMS, master scale) vs the music's loudest 2-6 kHz 5 ms level
  loudness    integrated LUFS vs the design's target, true peak, LRA, loudest momentary (time, share of the runtime),
              short-term curve, per-act medians when the cue file has acts, head and tail of the master
  coverage    cue events with no sound (unsounded or placed but silent); classes of action sounded only in part
  sync        impacts by attack / peak (-45..+75 ms), whooshes and peak-aligned swells by their apex (+-45 ms), risers
              and end-aligned swells by their audible end vs the hit (-250..+30 ms); measured norms of professional
              launch films. A deviation that exactly one layer's rule offset explains is listed apart as deliberate
  density     events per second, the most reused samples, "machine-gun" rules (one sample, no pitch or level change)
  listen      --listen: earshot timeline + sheet + critique of the master; --realness: earshot realness of every source
Thresholds and where they come from: references/process.md. Writes --out (JSON) and prints a digest. Exit code 1 when
a section fails (a click at a placement edge in the master, a silent effect, stems of pure digital silence, the
loudness gate), 2 on missing or mismatched inputs (one line says which flag to pass), else 0.
"""
import argparse, json, math, os, re, sys

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import master as M  # noqa: E402
import mix  # noqa: E402  (read-only: the design loader, the sample library, envelope helpers)

SR = mix.SR
LONG = ('whoosh', 'riser', 'swell', 'suck')
FC = 10 ** (np.arange(20, 42) / 10)  # 1/3-octave centres 100 Hz .. 12.6 kHz
BLO, BHI = FC * 2 ** (-1 / 6), FC * 2 ** (1 / 6)
SILENT_DB = -70.0  # an event whose peak (master scale) is below this makes no sound: a trim that missed the recording
CAP_DB = 60.0      # margins against a (near-)silent bed are reported as at most +60 dB


class InputError(Exception):
    pass


def db(v):
    return 20 * math.log10(max(float(v), 1e-12))


def pdb(p):
    return 10 * math.log10(max(float(p), 1e-20))


def fmt_t(t):
    m = int(t // 60)
    return f'{m}:{t - 60 * m:05.2f}'


def say(s):
    print(str(s).encode('ascii', 'replace').decode())


def lufs(x):
    """integrated LUFS, or None for audio shorter than one 400 ms block or with no gated signal (silence)"""
    if x is None or len(x) < int(0.4 * SR):
        return None
    v = M.integrated(x)
    return v if math.isfinite(v) else None


def clean(o):
    """JSON-safe: numpy scalars to Python, NaN / inf to None"""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, a):
        self.root = os.path.abspath(a.root)
        mix.ROOT = self.root
        mix.CACHE = os.path.join(self.root, a.cache) if a.cache else os.path.join(self.root, 'out', 'audio', 'cache')
        R = self.R = lambda p: os.path.normpath(p if os.path.isabs(p) else os.path.join(self.root, p))

        def need(path, what, flag):
            if not os.path.exists(R(path)):
                raise InputError(f'{what} not found: {R(path)} ({flag})')
            return R(path)

        need(a.design, 'design file', 'pass --design')
        self.design = mix.load_design(a.design, a.only)
        self.report = json.load(open(need(a.report, 'mix report', 'run mix.py first or pass --report'), encoding='utf-8'))
        self.trace = json.load(open(R(a.trace))) if a.trace and os.path.exists(R(a.trace)) else None
        self.stems = {}
        for k in ('sfx', 'amb', 'music'):
            p = need(os.path.join(a.stems, k + '.wav'), f'stem {k}.wav', 'pass the --stems folder mix.py wrote')
            self.stems[k] = sf.read(p, dtype='float32', always_2d=True)[0]
        self.N = len(self.stems['sfx'])
        if {len(v) for v in self.stems.values()} != {self.N}:
            raise InputError(f'the stems in {R(a.stems)} have different lengths: re-render them with one mix.py run')
        if a.master and getattr(a, 'master_given', False):
            need(a.master, 'master', 'pass --master none to skip it')
        self.master = (sf.read(R(a.master), dtype='float64', always_2d=True)[0]
                       if a.master and os.path.exists(R(a.master)) else None)
        if self.master is not None and abs(len(self.master) - self.N) > 1:
            raise InputError(f'the master {R(a.master)} ({len(self.master) / SR:.3f} s) does not match the stems '
                             f'({self.N / SR:.3f} s): pass the --master rendered with these stems, or --master none')
        cp = a.cues or self.design.get('cues') or 'out/cues.json'
        c = json.load(open(R(cp))) if os.path.exists(R(cp)) else {'fps': 60, 'events': []}
        self.cues = c[0] if isinstance(c, list) else c
        self.fps = self.cues.get('fps', 60)
        m = self.design.get('master') or {}
        self.target = m.get('lufs', -14)
        pre = sum(self.stems[k].astype(np.float64) for k in self.stems)
        self.pre_lufs = lufs(pre)
        self.master_lufs = lufs(self.master)
        both = self.master_lufs is not None and self.pre_lufs is not None
        self.G = self.master_lufs - self.pre_lufs if both else 0.0  # no master (or silence): stem scale
        self.scale = ('master' if both else 'stem') if self.master is not None else 'stem'
        self.no_bed = not (np.any(self.stems['music']) or np.any(self.stems['amb']))
        self.lib = None
        self.only = a.only


def bus_curve(design, bus, n):
    b = (design.get('buses') or {}).get(bus) or {}
    c = mix.auto_curve(b.get('auto'), n)
    return (np.ones(n) if c is None else c.astype(np.float64)) * mix.db2a(b.get('gain_db', 0.0))


def build_layers(ctx):
    """every sfx-bus placement rebuilt as it sits in the stem (sample, pitch, gain, pan, bus ride), grouped into events"""
    recs = [a for a in ctx.trace if 'sample' in a]
    pl = ctx.report.get('placements', [])
    if len(recs) != len(pl):
        return None, f'trace ({len(recs)} placements) and report ({len(pl)}) disagree: re-run mix.py with --trace'
    ctx.lib = ctx.lib or mix.Lib(ctx.design)
    places = iter(ctx.design.get('place') or [])
    rules = {str(r.get('name', i)): r for i, r in enumerate(ctx.design.get('rules') or [])}
    curve = bus_curve(ctx.design, 'sfx', ctx.N)
    out, bad = [], 0
    for a, p in zip(recs, pl):
        if p['kind'] == 'place':
            pp = next(places)
            pitch, off, align, cue, rule = pp.get('pitch', 0.0), pp.get('offset', 0.0), pp.get('align', 'attack'), pp['at'], 'place'
            ok = pp.get('qa_ok')
        else:
            r = rules.get(str(p.get('rule')), {})
            pitch, off, align, rule = p.get('pitch', 0.0), r.get('offset', 0.0), p.get('align', 'attack'), str(p.get('rule'))
            cue, ok = p['t'] - off, r.get('qa_ok')
        x = ctx.lib.get(a['sample'], pitch)
        if len(x) != a['n']:
            bad += 1
            continue
        if a['bus'] != 'sfx':
            continue
        s0, s1 = max(0, a['start']), min(ctx.N, a['start'] + a['n'])
        if s1 <= s0:
            continue
        seg = x[s0 - a['start']: s1 - a['start']].astype(np.float64) * a['gain']
        if a['pan']:
            seg = seg * np.array(mix.pan_gains(a['pan']))
        seg = seg * curve[s0:s1, None]
        out.append({'sample': a['sample'], 'rule': rule, 'kind': p['kind'], 'event': p.get('event'), 'cue': cue,
                    'offset': off, 'align': align, 'start': a['start'], 'n': a['n'], 'a': s0, 'b': s1, 'sig': seg, 'qa_ok': ok,
                    'att': a['start'] + mix.env_index(x, 'attack'), 'peak_db': db(np.abs(seg).max())})
    ctx.layers = out
    groups, by_cue = [], {}
    for L in sorted(out, key=lambda L: L['att']):
        if L['kind'] != 'place':  # layers of one cue event share its exact cue time
            k = (L['event'], round(L['cue'], 4))
            if k in by_cue:
                by_cue[k]['layers'].append(L)
            else:
                by_cue[k] = {'att0': L['att'], 'layers': [L]}
                groups.append(by_cue[k])
            continue
        for g in groups[::-1][:12]:  # free placements: layers whose attacks sit within 140 ms
            if g.get('place') and abs(L['att'] - g['att0']) < 0.14 * SR:
                g['layers'].append(L)
                break
        else:
            groups.append({'place': True, 'att0': L['att'], 'layers': [L]})
    for g in groups:
        ls = g['layers']
        g['a'], g['b'] = min(L['a'] for L in ls), max(L['b'] for L in ls)
        g['sig'] = np.zeros((g['b'] - g['a'], 2))
        for L in ls:
            g['sig'][L['a'] - g['a']: L['b'] - g['a']] += L['sig']
        g['cue'] = min(L['cue'] for L in ls)
        g['loud'] = max(ls, key=lambda L: L['peak_db'])
        g['kind'] = g['loud']['kind']
        g['label'] = (ls[0]['event'] or 'place:' + ls[0]['sample'])
        g['long'] = any(L['kind'] in LONG or L['align'] == 'peak' for L in ls)
        g['builds'] = any(L['kind'] in ('riser', 'swell') for L in ls)
        g['qa_ok'] = next((L['qa_ok'] for L in ls if L['qa_ok']), None)
        g['peak_db'] = db(np.abs(g['sig']).max()) + ctx.G
        g['silent'] = g['peak_db'] < SILENT_DB
    note = f'{bad} placement(s) could not be rebuilt (sample length differs from the trace)' if bad else None
    return groups, note


# ------------------------------------------------------------------ clicks
HP14 = signal.butter(4, 14000, 'high', fs=SR, output='sos')


def detect_clicks(x):
    """> 14 kHz samples > 9x the RMS of their 20 ms and > 6x their 4 ms neighbourhood; conc >= 0.8 of the burst's
    > 14 kHz energy inside 1 ms = a click (a cut), else a designed attack"""
    out = []
    for ch in range(x.shape[1]):
        hp = signal.sosfilt(HP14, x[:, ch].astype(np.float64))
        e, e2 = np.abs(hp), hp ** 2
        loc = np.sqrt(signal.fftconvolve(e2, np.ones(960) / 960, mode='same').clip(0)) + 1e-5
        near = np.sqrt(signal.fftconvolve(e2, np.ones(192) / 192, mode='same').clip(0)) + 1e-5
        last = -10 ** 9
        for i in np.nonzero((e / loc > 9) & (e / near > 6) & (e > 0.01))[0]:
            if i - last > SR * 0.01:
                conc = float(e2[max(0, i - 24):i + 24].sum() / (e2[max(0, i - 240):i + 240].sum() + 1e-15))
                out.append({'t': round(i / SR, 4), 'i': int(i), 'ch': ch, 'level_db': round(db(e[i]), 1),
                            'conc': round(conc, 2), 'kind': 'click' if conc >= 0.8 else 'attack'})
            last = i
    out.sort(key=lambda c: (c['t'], c['ch']))
    return [c for k, c in enumerate(out) if k == 0 or c['t'] - out[k - 1]['t'] > 0.001]


def owners(trace, i, bus=None, win=960):
    """trace records around sample i, edge-near (within 2 ms of a start / end) first"""
    res = []
    for a in trace or []:
        if bus and a['bus'] != bus:
            continue
        s, e = a['start'], a['start'] + a['n']
        if s - win <= i <= e + win:
            d = min(abs(i - s), abs(i - e))
            where = 'start' if abs(i - s) <= 96 else 'end' if abs(i - e) <= 96 else 'body'
            what = a.get('sample') or a.get('name') or a.get('label') or f"{a.get('kind', a['bus'])} at {s / SR:.3f} s"
            res.append((where == 'body', d, {'bus': a['bus'], 'what': what,
                                             'rule': a.get('rule', a.get('kind')), 'where': where,
                                             'from_start_ms': round((i - s) / SR * 1000, 1),
                                             'from_end_ms': round((i - e) / SR * 1000, 1)}))
    return [r for _, _, r in sorted(res, key=lambda r: (r[0], r[1]))[:3]]


def hf_owner(ctx, i, ch):
    """the rebuilt sfx layer carrying the > 14 kHz burst at sample i (its level in the stem, where in the layer)"""
    best = None
    for L in getattr(ctx, 'layers', None) or []:
        if not L['a'] - 480 <= i < L['b'] + 480:
            continue
        j = i - L['a']
        lo, hi = max(0, j - 480), min(len(L['sig']), j + 480)
        if hi - lo < 64:
            continue
        hp = signal.sosfilt(HP14, L['sig'][lo:hi, ch])
        c = j - lo
        v = float(np.abs(hp[max(0, c - 24):c + 24]).max()) if 0 <= c < len(hp) else 0.0
        if best is None or v > best[0]:
            off = i - L['start']
            where = 'start' if abs(off) <= 96 else 'end' if abs(i - L['start'] - L['n']) <= 96 else 'body'
            best = (v, {'bus': 'sfx', 'what': L['sample'], 'rule': L['rule'], 'where': where,
                        'from_start_ms': round(off / SR * 1000, 1), 'level_db': round(db(v), 1)})
    return best


def check_clicks(ctx):
    sets = [('master', ctx.master)] if ctx.master is not None else []
    sets += [(k, ctx.stems[k]) for k in ('sfx', 'amb', 'music')]
    items, counts = [], {}
    for name, x in sets:
        found = detect_clicks(x)
        counts[name] = {'clicks': sum(c['kind'] == 'click' for c in found), 'attacks': sum(c['kind'] == 'attack' for c in found)}
        for c in found:
            if c['kind'] != 'click':
                continue
            own = hf_owner(ctx, c['i'], c['ch']) if name in ('master', 'sfx') else None
            scale = 10 ** (-ctx.G / 20) if name == 'master' else 1.0  # master level back to stem scale
            if own and own[0] >= 0.3 * 10 ** (c['level_db'] / 20) * scale:
                ow = [own[1]]
            else:
                ow = owners(ctx.trace, c['i'], None if name == 'master' else name)
            items.append({'in': name, **{k: v for k, v in c.items() if k != 'i'}, 'owners': ow,
                          'at_edge': bool(ow) and ow[0]['where'] in ('start', 'end')})
    mc = [c for c in items if c['in'] == 'master']
    edge = [c for c in items if c['at_edge']]
    sc = len(items) - len(mc)
    status = 'fail' if any(c['in'] == 'master' for c in edge) else 'warn' if items else 'ok'
    summ = (f'{len(mc)} click(s) in the master, {sc} in the stems; {len(edge)} at a placement edge (a cut: fix the trim / '
            'fade)' + (', the rest inside a recording (its own crisp transient or a defect in the source)' if items else '')
            + ('' if items else ' (designed attacks: ' + ', '.join(f"{k} {v['attacks']}" for k, v in counts.items()) + ')'))
    return {'status': status, 'summary': summ, 'counts': counts, 'items': items}


# ------------------------------------------------------------------ edges
def rms(z):
    return float(np.sqrt((np.asarray(z, np.float64) ** 2).mean())) if len(z) else 0.0


def sample_edges(x, raw_pre, raw_post, rawpk):
    """unfaded head / tail 1 ms RMS vs the sample's peak, the source 20 ms outside the trim vs the trim's peak, and a
    second hit (a 5 ms envelope jump > 12 dB to within 18 dB of the peak) in the last 40 %"""
    m = np.abs(x).max(axis=1)
    pk = float(m.max()) or 1e-9
    w = 240
    n = len(m) // w
    e = m[:n * w].reshape(n, w).max(axis=1) if n else np.zeros(1)
    late = None
    for k in range(max(1, int(n * 0.6)), n):
        if e[k] > pk * 0.12 and e[k] > 4 * e[k - 1]:
            late = round(k * w / SR, 3)
            break
    return {'head_db': round(db(rms(x[:48]) / pk), 1), 'tail_db': round(db(rms(x[-48:]) / pk), 1),
            'pre20_db': round(db(rms(raw_pre) / rawpk), 1), 'post20_db': round(db(rms(raw_post) / rawpk), 1),
            'attack_ms': round(mix.env_index(x, 'attack') / SR * 1000, 1), 'late_onset': late}


def judge_edges(r, fi, fo, trimmed_start):
    """(defects, notes): a defect is audible as a click or a chop; a note is a craft suggestion"""
    bad, notes = [], []
    if trimmed_start and r['pre20_db'] > -20:
        msg = f"the trim cuts into a sound already playing (source 20 ms before at {r['pre20_db']} dB re peak)"
        if fi < 0.005 and (r['attack_ms'] > 10 or r['pre20_db'] > -10):  # exposed: no attack right after to cover it
            bad.append(msg + f', fade_in only {fi * 1000:.0f} ms')
        else:
            notes.append(msg + ', softened by the fade-in')
    elif r['head_db'] > -30:
        notes.append(f"the attack starts in the first ms (head {r['head_db']} dB): the {fi * 1000:.0f} ms fade-in dulls it; "
                     'start the trim 3-5 ms earlier')
    if r['tail_db'] > -35 and r['post20_db'] > -35:
        msg = f"the sound continues after the trim (tail {r['tail_db']} dB, source after {r['post20_db']} dB)"
        if fo < 0.03:
            bad.append(msg + f': chopped with a {fo * 1000:.0f} ms fade')
        else:
            notes.append(msg + ': decay faded early')
    if r['late_onset'] is not None:
        notes.append(f"a second hit inside the sample at {r['late_onset']} s (fine for a rattle, wrong for a single hit)")
    return bad, notes


def check_splices(ctx):
    """music segment boundaries: a segment end that chops a note still sounding into quieter material, holes and overlaps
    between segments, segments that begin inside a sustained sound"""
    mu = ctx.design.get('music')
    if not mu or not mu.get('segments'):
        return []
    ratio = mu.get('target_bpm', 120) / mu['src_bpm'] if mu.get('stretch', 'rubberband') != 'none' else 1.0
    x = mix.stretched(mu['file'], ratio)
    k = 1.0 / ratio
    w = int(0.05 * SR)

    def lv(i0):
        return db(rms(x[max(0, i0):max(0, i0 + w)]))

    segs = sorted(mu['segments'], key=lambda sg: sg['at'])
    dur = ctx.N / SR
    out = []
    for n, sg in enumerate(segs):
        s0, e0 = int(round(sg['src_from'] * k * SR)), int(round(sg['src_to'] * k * SR))
        end_t = sg['at'] + (sg['src_to'] - sg['src_from']) * k
        nxt = segs[n + 1] if n + 1 < len(segs) else None
        out_l, cut_l = lv(e0 - w), lv(e0)
        in_l = lv(int(round(nxt['src_from'] * k * SR))) if nxt else -120.0
        item = {'segment': sg.get('label') or n, 'film_end': round(end_t, 3), 'out_db': round(out_l, 1),
                'cut_off_db': round(cut_l, 1), 'next_in_db': round(in_l, 1) if nxt else None, 'issues': [], 'notes': []}
        fo = sg.get('fade_out', 0.06)
        if not sg.get('tail_fade') and fo < 0.1 and end_t < dur - 0.05 and cut_l > out_l - 6 and cut_l > in_l + 6:
            msg = (f'chops the music at {end_t:.3f} s: the source keeps sounding at {cut_l:.0f} dBFS and the next '
                   f'material enters at {in_l:.0f} (fade_out {fo * 1000:.0f} ms)')
            if sg.get('qa_ok'):
                item['notes'].append(f"acknowledged ({sg['qa_ok']}): {msg}")
            else:
                item['issues'].append(msg)
        if nxt:
            gap = nxt['at'] - end_t
            if gap > 0.02:
                item['notes'].append(f'{gap * 1000:.0f} ms of no music before {nxt["at"]:.3f} s (deliberate silence?)')
            elif gap < -0.1:
                item['notes'].append(f'overlaps the next segment by {-gap * 1000:.0f} ms (two passages at once)')
        if s0 > w and lv(s0 - w) > lv(s0) - 3:
            item['notes'].append(f"begins inside a sustained sound (the source before {sg['src_from']} s is as loud): "
                                 'no fresh attack on the splice')
        if item['issues'] or item['notes']:
            out.append(item)
    return out


def check_edges(ctx):
    items = []
    used = {p.get('sample') for p in ctx.report.get('placements', [])}
    lib = ctx.lib = ctx.lib or mix.Lib(ctx.design)
    for name, s in (ctx.design.get('samples') or {}).items():
        if name not in used:
            continue
        raw = mix.decode(s['file'])
        a = int(s.get('start', 0) * SR)
        b = int(s['end'] * SR) if s.get('end') is not None else len(raw)
        fi, fo = s.get('fade_in', 0.002), s.get('fade_out', 0.012)
        trim_pk = float(np.abs(raw[a:b]).max()) if b - a > 0 else 0.0
        if db(trim_pk) < -60:  # nothing to judge: the trim missed the sound
            why = ('empty (start >= end or past the file end)' if b - a <= 0 else 'digital silence' if trim_pk == 0
                   else f'near-silent (peak {db(trim_pk):.0f} dBFS)')
            items.append({'what': 'sample', 'name': name, 'file': os.path.basename(s['file']), 'start': s.get('start', 0),
                          'end': s.get('end'), 'trim_peak_dbfs': round(db(trim_pk), 1), 'fade_in': fi, 'fade_out': fo,
                          'issues': [f'the trim is {why}: it misses the sound (file length {len(raw) / SR:.3f} s)'],
                          'notes': []})
            continue
        x = lib._finish(raw[a:b], {**s, 'fade_in': 0, 'fade_out': 0})
        r = sample_edges(x, raw[max(0, a - 960):a], raw[b:b + 960], trim_pk)
        bad, notes = judge_edges(r, fi, fo, a > 0)
        if bad and s.get('qa_ok'):
            notes, bad = notes + [f'acknowledged ({s["qa_ok"]}): ' + b_ for b_ in bad], []
        if bad or notes:
            items.append({'what': 'sample', 'name': name, 'file': os.path.basename(s['file']), 'start': s.get('start', 0),
                          'end': s.get('end'), **r, 'fade_in': fi, 'fade_out': fo, 'issues': bad, 'notes': notes})
    for name, s in (ctx.design.get('slices') or {}).items():
        if not any(u and u.startswith(name + '#') for u in used):
            continue
        try:
            sl = lib._slice({**s, 'fade_in': 0, 'fade_out': 0})
        except Exception as e:  # noqa: BLE001
            items.append({'what': 'slices', 'name': name, 'issues': [f'cannot slice: {e}'], 'notes': []})
            continue
        fi, fo = s.get('fade_in', 0.002), s.get('fade_out', 0.02)
        bad, notes = set(), set()
        for x, _ in sl:
            r = sample_edges(x, x[:0], x[:0], float(np.abs(x).max()) or 1e-9)
            r['post20_db'] = r['tail_db']  # a slice ends where the next hit begins: its tail level is what is cut
            b_, n_ = judge_edges(r, fi, fo, False)
            bad |= {m.split(' (')[0] for m in b_}
            notes |= {m.split(' (')[0] for m in n_}
        if bad or notes:
            items.append({'what': 'slices', 'name': name, 'n': len(sl), 'issues': sorted(bad), 'notes': sorted(notes)})
    cut = []
    for a in ctx.trace or []:
        if a['bus'] != 'music' and (a['start'] < 0 or a['start'] + a['n'] > ctx.N):
            cut.append({'bus': a['bus'], 'what': a.get('sample') or a.get('name') or a.get('label'),
                        'start': round(a['start'] / SR, 3), 'end': round((a['start'] + a['n']) / SR, 3)})
    splices = check_splices(ctx)
    n_bad = sum(1 for it in items + splices if it['issues'])
    n_note = sum(1 for it in items + splices if it['notes'] and not it['issues'])
    summ = (f'{n_bad} defect(s) (abrupt entries, chopped decays, chopped music), {n_note} with craft notes only (used '
            f'samples, slices, music splices); {len(cut)} placement(s) run past the film ends')
    return {'status': 'warn' if n_bad else 'ok', 'summary': summ,
            'items': sorted(items, key=lambda it: (not it['issues'], it['name'])), 'splices': splices,
            'cut_by_film_ends': cut}


# ------------------------------------------------------------------ audibility / harshness / sync
def bands(seg):
    w = signal.windows.tukey(len(seg), 0.25)[:, None]
    S = (np.abs(np.fft.rfft(seg * w, axis=0)) ** 2).sum(axis=1)
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    c = np.concatenate([[0], np.cumsum(S)])
    ia, ib = np.searchsorted(f, BLO), np.searchsorted(f, BHI)
    return c[ib] - c[ia] + 1e-30


def window_at(arr, a0, s, w):
    o = arr[max(0, s - a0): max(0, s - a0) + w]
    return np.pad(o, ((0, w - len(o)), (0, 0))) if len(o) < w else o


def read_metrics(ctx, eff, a, b, att, long):
    """m60 vs bed, m60r / m20 vs everything else (best band within 30 dB of the effect's strongest)"""
    bed = ctx.stems['music'][a:b].astype(np.float64) + ctx.stems['amb'][a:b]
    rest = ctx.stems['sfx'][a:b] - eff + bed
    e2 = (eff ** 2).sum(axis=1)
    c = np.concatenate([[0], np.cumsum(e2)])

    def best(w, s0, s1):
        s0 = max(a, s0)
        s1 = max(s0, min(b - w, s1))
        cand = np.arange(s0, s1 + 1, 48)
        en = [c[min(len(c) - 1, s - a + w)] - c[max(0, s - a)] for s in cand]
        return int(cand[int(np.argmax(en))])

    w60, w20 = int(0.06 * SR), int(0.02 * SR)
    s60 = best(w60, att - 240, (b - w60) if long else att + int(0.3 * SR))
    s20 = s60 if long else best(w20, att - 144, att + int(0.08 * SR))
    E, B, Rr = (bands(window_at(z, a, s60, w60)) for z in (eff, bed, rest))
    E20, R20 = bands(window_at(eff, a, s20, w20)), bands(window_at(rest, a, s20, w20))

    def mb(Ep, Bp):
        ok = Ep >= Ep.max() * 1e-3
        v = np.where(ok, np.clip(10 * np.log10(Ep / Bp), -CAP_DB, CAP_DB), -99)
        k = int(np.argmax(v))
        return round(float(v[k]), 1), int(round(FC[k]))

    m60, f60 = mb(E, B)
    m60r, _ = mb(E, Rr)
    m20, f20 = mb(E20, R20)
    read = max(m60r, m20)
    verdict = 'clear' if read >= 6 else 'ok' if read >= 3 else 'barely' if read >= 0 else 'masked'
    return {'vs_bed_db': m60, 'vs_bed_hz': f60, 'vs_all_db': m60r, 'first20_db': m20, 'first20_hz': f20,
            'read_db': read, 'verdict': verdict, 'covered_by': 'music/amb' if m60 < 3 else 'other effects'}


def check_audibility(ctx, groups, layers_too=False):
    rows = []
    for g in groups:
        att = min(L['att'] for L in g['layers'])
        r = {'t': round(att / SR, 3), 'label': g['label'], 'kind': g['kind'], 'n_layers': len(g['layers']),
             'rules': sorted({L['rule'] for L in g['layers']}), 'peak_db': round(max(g['peak_db'], -120), 1)}
        if g['silent']:
            r.update({'read_db': None, 'verdict': 'silent',
                      'samples': sorted({L['sample'] for L in g['layers']})})
        elif ctx.no_bed:
            r['verdict'] = 'not judged'
        else:
            r.update(read_metrics(ctx, g['sig'], g['a'], g['b'], att, g['long']))
        if g['qa_ok']:
            r['qa_ok'] = g['qa_ok']
        rows.append(r)
    cnt = {}
    for r in rows:
        cnt[r['verdict']] = cnt.get(r['verdict'], 0) + 1
    masked = [r for r in rows if r['verdict'] == 'masked' and not r.get('qa_ok')]
    silent = [r for r in rows if r['verdict'] == 'silent' and not r.get('qa_ok')]
    sil = (f'; {len(silent)} SILENT (peak under {SILENT_DB:.0f} dBFS: the trim missed the sound) at '
           + ', '.join(f"{r['t']:.2f} {r['label']} ({', '.join(r['samples'])})" for r in silent[:6]) if silent else '')
    if ctx.no_bed:
        return {'status': 'fail' if silent else 'skipped', 'counts': cnt, 'events': rows, 'items': silent,
                'summary': f'{len(rows)} events, not judged against a bed: this render has no music or ambience (a '
                           'solo render); judge audibility on the full mix' + sil}
    judged = sorted((r for r in rows if r['read_db'] is not None), key=lambda r: r['read_db'])
    res = {'status': 'fail' if silent else 'warn' if masked else 'ok', 'counts': cnt,
           'summary': f'{len(rows)} events: ' + ', '.join(f'{k} {cnt.get(k, 0)}' for k in ('clear', 'ok', 'barely', 'masked'))
           + sil + ('' if not masked else '; masked at ' + ', '.join(f"{r['t']:.2f} {r['label']} (by {r['covered_by']})"
                                                                   for r in masked[:6])),
           'items': (silent + judged)[:40], 'events': rows}
    if layers_too:
        dead = []
        for g in groups:
            if len(g['layers']) < 2 or g['silent']:
                continue
            for L in g['layers']:
                if L['peak_db'] + ctx.G < SILENT_DB:
                    dead.append({'t': round(L['att'] / SR, 3), 'rule': L['rule'], 'sample': L['sample'], 'read_db': None,
                                 'silent': True})
                    continue
                r = read_metrics(ctx, L['sig'], L['a'], L['b'], L['att'], g['long'])
                if r['read_db'] < 0:
                    dead.append({'t': round(L['att'] / SR, 3), 'rule': L['rule'], 'sample': L['sample'], 'read_db': r['read_db']})
        res['dead_layers'] = dead
        res['summary'] += f'; {len(dead)} layer(s) in multi-layer events add nothing measurable (read < 0 dB)'
    return res


def centroid(x):
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float((f * S).sum() / (S.sum() + 1e-30))


def check_harshness(ctx, groups):
    sos = signal.butter(4, [2000, 6000], 'bandpass', fs=SR, output='sos')
    sol = signal.butter(4, [80, 400], 'bandpass', fs=SR, output='sos')
    mu = signal.sosfilt(sos, ctx.stems['music'].astype(np.float64).mean(axis=1))
    e5 = np.convolve(mu ** 2, np.ones(240) / 240, 'same')
    has_music = e5.max() > 1e-10  # a solo render without the music: presence is not judged
    ref = pdb(e5.max()) + ctx.G if has_music else None
    rows = []
    for g in groups:
        att = min(L['att'] for L in g['layers'])
        x = g['sig'].mean(axis=1)
        pres = signal.sosfilt(sos, np.r_[np.zeros(480), x])[480:]
        lo = signal.sosfilt(sol, np.r_[np.zeros(480), x])[480:]
        i0 = max(0, att - g['a'] - 480)
        i1 = len(x) if g['long'] else min(len(x), att - g['a'] + int(0.08 * SR))
        if i1 - i0 < 240:
            continue
        pk = pdb(np.convolve(pres[i0:i1] ** 2, np.ones(240) / 240, 'valid').max()) + ctx.G
        lk = pdb(np.convolve(lo[i0:i1] ** 2, np.ones(240) / 240, 'valid').max()) + ctx.G
        m0, m1 = max(0, att - SR // 4), min(len(mu), att + SR // 4)
        r = {'t': round(att / SR, 3), 'label': g['label'], 'rules': sorted({L['rule'] for L in g['layers']}),
             'pres_db': round(pk, 1), 'pres_minus_low_db': round(pk - lk, 1), 'centroid_hz': int(centroid(x[i0:i1])),
             'music_pres_db': round(pdb((mu[m0:m1] ** 2).mean()) + ctx.G, 1),
             'over_music_max_db': round(pk - ref, 1) if has_music else None}
        r['issues'] = []
        if has_music and r['over_music_max_db'] > 6:
            r['issues'].append(f"2-6 kHz peak {r['pres_db']} dBFS, {r['over_music_max_db']:+.1f} dB over the music's loudest")
        r['bright'] = r['centroid_hz'] > 6500
        if g['qa_ok'] and r['issues']:
            r['qa_ok'], r['issues'] = g['qa_ok'], []
        rows.append(r)
    rows.sort(key=lambda r: -r['pres_db'])
    hot = [r for r in rows if r['issues']]
    bright = [r for r in rows if r['bright']]
    head = (f"{len(hot)} event(s) more than 6 dB over the music's loudest 2-6 kHz peak ({ref:.1f} dBFS)" if has_music else
            'no music in this render: presence vs the music not judged')
    return {'status': 'warn' if hot else 'ok', 'music_max_db': round(ref, 1) if has_music else None,
            'summary': (head + ('' if not hot else ': ' + ', '.join(f"{r['t']:.2f} {r['label']}" for r in hot[:6]))
                        + f"; note: {len(bright)} bright event(s) (centroid > 6.5 kHz, above launch-film effects)"),
            'items': hot[:20] + [r for r in rows if not r['issues']][:max(0, 10 - len(hot))],
            'bright': [{'t': r['t'], 'label': r['label'], 'centroid_hz': r['centroid_hz']} for r in bright]}


def sync_issue(test, r, s):
    """the sync verdict for a measured event shifted by s ms (a rule offset): None when it sits on its cue"""
    if test == 'end':  # a riser stops on (or 30-250 ms before) its hit
        e = r['end_ms'] - s
        return f"its audible end is {e:+d} ms from its hit (want -250..+30)" if not -300 <= e <= 60 else None
    if test == 'apex':
        e = r['apex_ms'] - s
        return f"apex {e:+d} ms from the cue (want +-45)" if abs(e) > 45 else None
    a, p = r['attack_ms'] - s, r['peak_ms'] - s
    return (f"attack {a:+d} ms (sound before picture)" if a < -45 else f"peak {p:+d} ms (lands late)" if p > 75 else None)


def check_sync(ctx, groups):
    """impacts by attack / peak, whooshes and peak-aligned swells by their apex, risers (and swells placed by their
    end or attack) by their audible end vs the hit; a deviation that one layer's rule offset explains is deliberate"""
    rows, deliberate, n = [], [], 0
    for g in groups:
        x = np.abs(g['sig']).max(axis=1)
        k = len(x) // 240
        if not k:
            continue
        align = g['loud']['align']
        test = (('apex' if align == 'peak' else 'end') if g['builds'] else None if align in ('start', 'end') else
                'apex' if g['long'] else 'impact')
        if test is None:  # placed by its start / end on purpose (not a build): no hit to judge
            continue
        n += 1
        pk = (int(np.argmax(x[:k * 240].reshape(k, 240).max(axis=1))) * 240 + 120 + g['a']) / SR
        p = np.convolve((g['sig'] ** 2).sum(axis=1), np.ones(2400) / 2400, 'same')
        apex = (int(np.argmax(p)) + g['a']) / SR
        att = g['loud']['att'] / SR
        r = {'t_cue': round(g['cue'], 3), 'label': g['label'], 'kind': g['kind'], 'test': test, 'align': align,
             'attack_ms': round((att - g['cue']) * 1000), 'peak_ms': round((pk - g['cue']) * 1000),
             'apex_ms': round((apex - g['cue']) * 1000), 'long': g['long']}
        if test == 'end':
            nz = np.nonzero(x > x.max() * 0.1)[0]  # the audible body: above 10 % of the peak (-20 dB)
            r['end_ms'] = round(((nz[-1] if len(nz) else 0) + g['a']) / SR * 1000 - g['cue'] * 1000)
        r['issue'] = sync_issue(test, r, 0)
        if not r['issue']:
            continue
        offs = sorted({(round(L['offset'] * 1000), L['rule']) for L in g['layers'] if abs(L['offset']) >= 0.03})
        why = next(((o, rule) for o, rule in offs if not sync_issue(test, r, o)), None)
        if why:
            r.update({'offset_s': why[0] / 1000, 'offset_rule': why[1],
                      'explained': f"{re.sub(r' [(].*', '', r['issue'])} = rule {why[1]}'s offset {why[0] / 1000:+.3f} s"})
            deliberate.append(r)
        elif g['qa_ok']:
            r['qa_ok'] = g['qa_ok']
            deliberate.append(r)
        else:
            if offs:
                r['offsets_tried_s'] = [o / 1000 for o, _ in offs]
            rows.append(r)
    return {'status': 'warn' if rows else 'ok', 'checked': n, 'items': rows, 'deliberate': deliberate,
            'summary': f'{n} events checked, {len(rows)} off their cue'
            + (': ' + ', '.join(f"{r['t_cue']:.2f} {r['label']} {r['issue']}" for r in rows[:5]) if rows else '')
            + f'; {len(deliberate)} more sit off the cue by exactly a rule offset or carry a qa_ok note (check those on '
            'the frames)'}


def check_density(ctx, groups):
    per_s = {}
    for g in groups:
        s = int(min(L['att'] for L in g['layers']) / SR)
        per_s[s] = per_s.get(s, 0) + 1
    pl = [p for p in ctx.report.get('placements', []) if p.get('kind') != 'place']
    use = {}
    for p in ctx.report.get('placements', []):
        use[p['sample']] = use.get(p['sample'], 0) + 1
    by = {}
    for p in pl:
        by.setdefault(str(p['rule']), []).append(p)
    rules, mg = [], []
    for k, v in by.items():
        if len(v) < 4:
            continue
        t = np.array(sorted(p['t'] for p in v))
        ioi = np.diff(t)
        g = np.array([p['gain_db'] for p in v])
        pt = np.array([p.get('pitch', 0) for p in v])
        r = {'rule': k, 'n': len(v), 'distinct_samples': len({p['sample'] for p in v}), 'ioi_s': round(float(ioi.mean()), 3),
             'ioi_cv': round(float(ioi.std() / max(ioi.mean(), 1e-9)), 2), 'gain_sd_db': round(float(g.std()), 2),
             'pitch_sd_st': round(float(pt.std()), 2)}
        rules.append(r)
        if r['distinct_samples'] == 1 and r['pitch_sd_st'] < 0.1 and r['gain_sd_db'] < 0.5:
            mg.append(r)
    n = ctx.N / SR
    busiest = sorted(per_s.items(), key=lambda kv: -kv[1])[:5]
    return {'status': 'warn' if mg else 'ok', 'events_per_s': [per_s.get(s, 0) for s in range(int(math.ceil(n)))],
            'busiest_s': busiest, 'most_used_samples': sorted(use.items(), key=lambda kv: -kv[1])[:8], 'rules': rules,
            'machine_gun': mg,
            'summary': (f'{len(groups)} effect events in {n:.0f} s ({len(groups) / n * 10:.1f} per 10 s); busiest seconds '
                        + ', '.join(f'{s}s:{c}' for s, c in busiest) + f'; {len(mg)} machine-gun rule(s)'
                        + (': ' + ', '.join(r['rule'] for r in mg) if mg else ''))}


# ------------------------------------------------------------------ loudness / coverage
def check_loudness(ctx):
    x = ctx.master
    if x is None:
        return {'status': 'skipped', 'summary': 'no master WAV (pass --master)'}
    if len(x) < int(0.4 * SR):
        return {'status': 'skipped', 'summary': f'the master is {len(x) / SR * 1000:.0f} ms long: loudness needs >= 400 ms'}
    if not np.any(x):
        return {'status': 'fail', 'lufs': None, 'summary': 'the master is digital silence (no gated loudness)'}
    I, tp, lr = M.integrated(x), M.true_peak_db(x), M.lra(x)
    tm, mom = M.blocks(x, 0.4, 0.1)
    ts, st = M.blocks(x, 3.0, 1.0)
    k = int(np.argmax(mom))
    dur = len(x) / SR
    head = db(np.abs(x[:240]).max())
    tail_rms = db(np.sqrt((x[-2400:] ** 2).mean()))
    tail_pk = db(np.abs(x[-480:]).max())
    r = {'lufs': round(I, 2), 'target': ctx.target, 'true_peak_dbtp': round(tp, 2), 'lra_lu': round(lr, 1),
         'premaster_lufs': round(ctx.pre_lufs, 2) if ctx.pre_lufs is not None else None, 'loudest_momentary': round(float(mom[k]), 1),
         'loudest_at': round(float(tm[k]), 2), 'loudest_frac': round(float(tm[k]) / dur, 2),
         'short_term_per_s': [round(float(v), 1) for v in st], 'head_5ms_peak_dbfs': round(head, 1),
         'tail_50ms_rms_dbfs': round(tail_rms, 1), 'tail_10ms_peak_dbfs': round(tail_pk, 1)}
    acts = ctx.cues.get('acts') or {}
    if acts:
        r['acts'] = {}
        for name, v in acts.items():
            t0, t1 = v['from'] / ctx.fps, (v['from'] + v['dur']) / ctx.fps
            sel = (tm >= t0) & (tm < t1)
            if sel.any():
                r['acts'][name] = {'from': round(t0, 2), 'momentary_median': round(float(np.median(mom[sel])), 1),
                                   'momentary_max': round(float(mom[sel].max()), 1)}
    fails, warns = [], []
    if abs(I - ctx.target) > 0.5:
        fails.append(f'integrated {I:.1f} LUFS, target {ctx.target}')
    if tp > -1.0:
        fails.append(f'true peak {tp:.1f} dBTP (> -1.0: AAC will clip)')
    if tail_rms > -45 or tail_pk > -60:
        fails.append(f'the end is not silent (last 50 ms {tail_rms:.0f} dBFS RMS, last 10 ms peak {tail_pk:.0f})')
    if head > -40:
        warns.append(f'first 5 ms peak {head:.0f} dBFS: starts mid-sound or clicks on sample 0')
    if lr < 4:
        warns.append(f'LRA {lr:.1f} LU: flat (launch films with an arc sit at 5-9 LU)')
    elif lr > 10:
        warns.append(f'LRA {lr:.1f} LU: wider than launch films (5-9 LU); quiet passages may vanish on phones')
    if r['loudest_frac'] > 0.85:
        warns.append(f'the loudest moment is at {r["loudest_frac"]:.0%} of the runtime: peak at the drop, resolve at the end')
    r['status'] = 'fail' if fails else 'warn' if warns else 'ok'
    r['summary'] = (f'{I:.1f} LUFS (target {ctx.target}), true peak {tp:.1f} dBTP, LRA {lr:.1f} LU, loudest momentary '
                    f'{mom[k]:.1f} LUFS at {fmt_t(tm[k])} ({r["loudest_frac"]:.0%} of the runtime)'
                    + ('; ' + '; '.join(fails + warns) if fails or warns else ''))
    return r


def stem_id(i):
    return re.sub(r'(:\d+)+$', '', str(i or ''))


def check_coverage(ctx, groups=None):
    """cue events with no sound: unsounded in the report, or placed but silent (every layer below SILENT_DB)"""
    ev = ctx.cues.get('events') or []
    uns = list(ctx.report.get('unsounded_events', []))
    quiet = [{'t': round(g['cue'], 3), 'kind': g['kind'], 'id': g['layers'][0]['event']}
             for g in groups or [] if g['silent'] and not g.get('place') and not g['qa_ok']]
    uns += quiet
    tot, miss = {}, {}
    for e in ev:
        k = (e['kind'], stem_id(e.get('id')))
        tot[k] = tot.get(k, 0) + 1
    for e in uns:
        k = (e['kind'], stem_id(e.get('id')))
        miss.setdefault(k, []).append(e['t'])
    partial = [{'kind': k[0], 'id': k[1], 'unsounded': len(v), 'of': tot.get(k, len(v)), 'at': v[:12]}
               for k, v in miss.items() if len(v) < tot.get(k, len(v))]
    silent = [{'kind': k[0], 'id': k[1], 'n': len(v), 'at': v[:12]} for k, v in miss.items() if len(v) >= tot.get(k, len(v))]
    return {'status': 'warn' if partial or quiet else 'ok', 'partial': partial, 'unsounded_classes': silent,
            'placed_but_silent': quiet,
            'summary': (('solo render (events of the other fragments are expected to have no sound): ' if ctx.only else '')
                        + f'{len(uns)} of {len(ev)} cue events have no sound'
                        + (f' ({len(quiet)} of them placed but silent: ' + ', '.join(f"{q['t']:.2f} {q['id']}" for q in quiet[:5])
                           + ')' if quiet else '') + f'; {len(partial)} class(es) of action sounded '
                        f'only in part' + (': ' + ', '.join(f"{p['kind']} {p['id']} ({p['unsounded']}/{p['of']} silent)"
                                                            for p in partial[:5]) if partial else '')
                        + (f'; classes never sounded: ' + ', '.join(f"{s['kind']} {s['id']} x{s['n']}" for s in silent[:8])
                           if silent else ''))}


# ------------------------------------------------------------------ earshot
def listen(ctx, a):
    try:
        from earshot import api
    except Exception as e:  # noqa: BLE001
        return {'status': 'skipped', 'summary': f'earshot is not importable here ({e}): pip install -e <earshot repo>'}
    out_dir = os.path.dirname(ctx.R(a.out))
    os.makedirs(out_dir, exist_ok=True)
    res, notes = {}, []
    video = ctx.R(a.video) if a.video else None
    if a.listen and ctx.master is not None:
        mp = ctx.R(a.master)
        cues_used = ctx.R(a.cues_sounded) if a.cues_sounded and os.path.exists(ctx.R(a.cues_sounded)) else None
        tl = {}
        steps = (('timeline', lambda: tl.setdefault('r', api.timeline(mp, video=video, cues=cues_used))),
                 ('sheet', lambda: api.sheet(mp, os.path.join(out_dir, 'qa_sheet.png'), video=video, cues=cues_used,
                                             tl=tl.get('r'))),
                 ('critique', lambda: api.critique(mp, video=video, use_ast=False, use_realness=False)))
        for name, call in steps:  # the critique is the fast scorecard: realness of the sources is --realness's job
            try:
                r = call()
                res[name] = {'summary': r.get('summary'), **({'png': r.get('png') or r.get('path')} if name == 'sheet' else {})}
                if name == 'timeline' and r.get('text'):
                    p = os.path.join(out_dir, 'qa_timeline.txt')
                    open(p, 'w', encoding='utf-8').write(r['text'])
                    res[name]['text_file'] = p
                if name == 'critique':
                    res[name].update({'verdict': r.get('verdict'), 'score': r.get('score'), 'actions': r.get('actions'),
                                      'off_range': [i for i in r.get('metrics') or [] if isinstance(i, dict)
                                                    and i.get('status') in ('low', 'high')]})
            except Exception as e:  # noqa: BLE001
                notes.append(f'{name}: {type(e).__name__}: {str(e)[:160]}')
    if a.realness:
        rows = []
        for f in ctx.report.get('files_used', []):
            try:
                r = api.realness(ctx.R(f))
                rows.append({'file': f, 'p_real': r.get('p_real'), 'verdict': r.get('verdict')})
            except Exception as e:  # noqa: BLE001
                notes.append(f'realness {os.path.basename(f)}: {str(e)[:160]}')
        rows.sort(key=lambda r: (r['p_real'] is None, r['p_real'] if r['p_real'] is not None else 1))
        res['realness'] = {'files': rows, 'not_real': [r for r in rows if (r['p_real'] < 0.5 if r['p_real'] is not None
                                                                           else 'synth' in str(r['verdict']).lower())]}
    short = lambda k, v: (v['summary'][:360].rsplit(' ', 1)[0] + f" ... (all of it: {v['text_file']})"
                          if k == 'timeline' and len(v['summary']) > 400 and v.get('text_file') else v['summary'])
    parts = [f"{k}: {short(k, v)}" for k, v in res.items() if k != 'realness' and v.get('summary')]
    if 'realness' in res:
        rr = res['realness']
        parts.append(f"realness: {len(rr['not_real'])} of {len(rr['files'])} sources judged not real" if rr['files'] else
                     'realness: not available (' + (notes[0] if notes else 'no sources') + ')')
    if notes:
        parts.append(f'{len(notes)} earshot call(s) failed (see notes)')
    bad = bool(res.get('realness', {}).get('not_real'))
    return {'status': 'warn' if bad or notes else 'ok', 'summary': '; '.join(parts) or 'nothing ran', **res, 'notes': notes}


# ------------------------------------------------------------------ run
def run_qa(a):
    ctx = Ctx(a)
    groups, note = build_layers(ctx) if ctx.trace is not None else (None, None)
    live = [g for g in groups if not g['silent']] if groups else groups
    sec = {'clicks': check_clicks(ctx), 'loudness': check_loudness(ctx), 'coverage': check_coverage(ctx, groups)}
    if ctx.trace is None:
        msg = 'needs the trace: run mix.py with --trace out/qa/adds.json'
        for k in ('edges', 'audibility', 'harshness', 'sync', 'density'):
            sec[k] = {'status': 'skipped', 'summary': msg}
    else:
        sec['edges'] = check_edges(ctx)
        if groups is None:
            for k in ('audibility', 'harshness', 'sync', 'density'):
                sec[k] = {'status': 'skipped', 'summary': note}
        else:
            sec['audibility'] = check_audibility(ctx, groups, a.layers)
            sec['harshness'] = check_harshness(ctx, live)
            sec['sync'] = check_sync(ctx, live)
            sec['density'] = check_density(ctx, live)
    if a.listen or a.realness:
        sec['listen'] = listen(ctx, a)
    if not any(np.any(v) for v in ctx.stems.values()):
        sec['mix'] = {'status': 'fail', 'summary': 'every stem is digital silence: nothing was mixed (check the trims with '
                                                   'mix.py --dump, the gains and the cue matches)'}
    order = {'fail': 0, 'warn': 1, 'ok': 2, 'skipped': 3}
    worst = min((s['status'] for s in sec.values()), key=lambda s: order.get(s, 9))
    bad = [k for k, s in sec.items() if s['status'] in ('fail', 'warn')]
    skipped = [k for k, s in sec.items() if s['status'] == 'skipped']
    summary = (f"QA {worst.upper()}: " + (', '.join(f"{k} {sec[k]['status']}" for k in bad) if bad else
                                         'every check that ran passed' if skipped else 'every check passed')
               + (f" (not run: {', '.join(skipped)})" if skipped else '')
               + (f". Master gain over the stems {ctx.G:+.1f} dB (levels below are at master scale)." if ctx.scale == 'master'
                  else '. Master or premaster silent / too short: levels below are at stem scale.' if ctx.master is not None
                  else '. No master: levels below are at stem scale.'))
    return {'summary': summary, 'status': worst, 'note': note, 'sections': sec,
            'inputs': {'design': a.design, 'report': a.report, 'trace': a.trace, 'stems': a.stems, 'master': a.master,
                       'only': a.only}}


def digest(res):
    lines = [res['summary']]
    for k, s in res['sections'].items():
        lines.append(f"[{s['status'].upper():7s}] {k}: {s['summary']}")
        if k == 'clicks':
            for c in s.get('items', [])[:6]:
                o = c['owners'][0] if c['owners'] else {}
                lines.append(f"    {c['in']} {c['t']:.3f} ({fmt_t(c['t'])}) ch{c['ch']} conc {c['conc']}: "
                             f"{o.get('bus', '?')}:{o.get('what', '?')} [{o.get('rule', '')}] at its {o.get('where', '?')}")
        elif k == 'edges':
            for it in [i for i in s.get('items', []) + s.get('splices', []) if i['issues']][:8]:
                lines.append(f"    {it.get('what', 'music')} {it.get('name', it.get('segment'))}: {'; '.join(it['issues'])}")
        elif k == 'audibility':
            for r in s.get('items', [])[:6]:
                if r['verdict'] == 'silent':
                    lines.append(f"    {r['t']:.3f} ({fmt_t(r['t'])}) {r['label']}: SILENT, peak {r['peak_db']} dBFS "
                                 f"[{', '.join(r['samples'])[:80]}]")
                elif r['verdict'] in ('masked', 'barely'):
                    lines.append(f"    {r['t']:.3f} ({fmt_t(r['t'])}) {r['label']}: read {r['read_db']} dB ({r['verdict']}), "
                                 f"vs bed {r['vs_bed_db']} dB @ {r['vs_bed_hz']} Hz [{', '.join(r['rules'])[:80]}]")
        elif k == 'harshness':
            for r in [r for r in s.get('items', []) if r['issues']][:5]:
                lines.append(f"    {r['t']:.3f} ({fmt_t(r['t'])}) {r['label']}: {'; '.join(r['issues'])} [{', '.join(r['rules'])[:70]}]")
        elif k == 'sync':
            for r in s.get('items', [])[:6]:
                lines.append(f"    cue {r['t_cue']:.3f} ({fmt_t(r['t_cue'])}) {r['label']}: {r['issue']}")
            for r in s.get('deliberate', [])[:6]:
                lines.append(f"    cue {r['t_cue']:.3f} ({fmt_t(r['t_cue'])}) {r['label']}: deliberate, "
                             + (r.get('explained') or f"{r['issue']} (qa_ok: {r.get('qa_ok')})"))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default='.', help='project root: relative paths resolve against it')
    ap.add_argument('--design', default='audio/sounddesign.json')
    ap.add_argument('--report', help='default out/qa/mix_report.json (--only: out/audio-review/solo/<only>_report.json)')
    ap.add_argument('--trace', default='out/qa/adds.json', help='mix.py --trace output (attribution, layers)')
    ap.add_argument('--stems', help='default out/audio/stems (--only: out/audio-review/solo/<only>_stems)')
    ap.add_argument('--master', help='default out/audio/soundtrack.wav (--only: none); "none" skips it')
    ap.add_argument('--cues', help='cue file (default: the design\'s "cues", else out/cues.json)')
    ap.add_argument('--cues-sounded', default='out/qa/cues_sounded.json', help='for earshot timeline (--listen)')
    ap.add_argument('--only', help='the same --only the mix was rendered with')
    ap.add_argument('--cache', help='decode cache (default <root>/out/audio/cache, as mix.py)')
    ap.add_argument('--video', help='the film, for earshot (--listen)')
    ap.add_argument('--out', default='out/qa/qa.json')
    ap.add_argument('--layers', action='store_true', help='also measure every layer of multi-layer events (dead layers)')
    ap.add_argument('--listen', action='store_true', help='earshot timeline + sheet + critique of the master')
    ap.add_argument('--realness', action='store_true', help='earshot realness of every source recording used')
    ap.add_argument('--json', action='store_true', help='print the full result as JSON')
    a = ap.parse_args(argv)
    base = 'out/audio-review/solo/' + re.sub(r'[^A-Za-z0-9._-]+', '_', a.only) if a.only else None
    a.report = a.report or (base + '_report.json' if base else 'out/qa/mix_report.json')
    a.stems = a.stems or (base + '_stems' if base else 'out/audio/stems')
    a.master_given = bool(a.master) and a.master.lower() != 'none'
    a.master = None if (a.master or '').lower() == 'none' else a.master or (None if base else 'out/audio/soundtrack.wav')
    out = os.path.normpath(a.out if os.path.isabs(a.out) else os.path.join(os.path.abspath(a.root), a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        res = run_qa(a)
    except InputError as e:
        say(f'qa.py: {e}')
        return 2
    full = clean(res)
    json.dump(full, open(out, 'w'), indent=1)
    say(json.dumps(full, indent=1) if a.json else digest(res) + f'\nwrote {out}')
    return 1 if res['status'] == 'fail' else 0


if __name__ == '__main__':
    sys.exit(main())
