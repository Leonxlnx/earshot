"""earshot sfx: one bank sound by id, aimed at the picture, and a model-free audit of any sound effect.

    earshot sfx list [--category C] [--kind K] [--json]
    earshot sfx render ID -o out.wav [--seed N] [--dur S] [--apex S] [--pitch NOTE|HZ] [--peak DB] [--weight W] [--json]
    earshot sfx check FILE [FILE ...] [--category C] [--role R] [--json]

    from earshot.forge import sfx
    sfx.check(y)                       # (n,) / (n, 2) float at 48 kHz, or a path: {'ok', 'warnings', 'notes', ...}
    sfx.rising_glide(y), sfx.layers(y)

`list` / `render` are the bank (earshot.forge.bank; knobs: earshot.forge.knobs). `check` measures, it never listens:
  silent         nothing to hear: a sample peak under SILENT_DBFS (-60 dBFS), or less than SILENT_MS (5 ms) within
                 40 dB of its peak (1 ms RMS frames; the shortest bank sound, pro_hover_tick/air, holds 9 ms): digital
                 silence, or a take faded to nothing
  clicks, edge   polish.clicks: isolated discontinuities inside the sound; a first / last sample off zero
  clipping       flat tops at the peak, samples over full scale
  dc             an offset (|mean| > -40 dB re peak)
  ring           a metallic comb in the tail (polish.ring z > 15)
  truncated      a tail cut while still decaying (above -45 dB); a note only for the categories whose stop is the
                 design (polish's gated ones: riser, pro_riser, pro_zip, glitch, pro_glitch, pro_impact) and for any
                 sound that ends on its loudest moment (a riser, a reverse swell)
  silence        digital silence inside the sound (a note for glitches)
  tail           a tail that rises again by more than 6 dB (a note: a second event, a roll, a double member)
  rising_glide   the cheap "synth whip" tell: a tonal sound whose pitch climbs more than GLIDE_ST (4) semitones over
                 at least 50 ms. rising_glide() tracks the strongest partial in 21 ms frames every 5 ms where it holds
                 at least half of the frame's 80 Hz - 10 kHz energy within +-3 semitones (a tone, also a fast smeared
                 sweep; a noise band is not), splits the track where it jumps more than 3 semitones (a partial
                 swap), and reports the largest climb of the 5-frame median above its running minimum. Measured on
                 the 183 forge v2 finals: it flags pro_zip up / up_hi / arc (+20..64 st), the zip and sweep leads of
                 logo_sting zip_ping and pro_logo_sting sweep_hit (+25, +29 st), and the bubbles pop/2 and pro_pop
                 bloop (+6 st over 55-75 ms, a physical bubble); falling sweeps and steady bells read 0. For the
                 drip category a climb a bubble makes (recipes/drip.allows: at most 14 st, starting within the first
                 60 ms, done within 250 ms) is a note, not a warning: a drop's pitch rises by design (earshot 0.4)
  stacked,       for a hit (role hit / land): layers() counts separate attacks in a 25 ms RMS envelope (peaks within
  pre_swell      6 dB of the loudest, at least 10 dB of dip between them, in the first 0.5 s) and the pre-swell (from
                 the first frame within 30 dB of the peak to the first within 6 dB): two attacks, or more than
                 PRE_SWELL_MS (100) of build before the hit, read as a stacked sting where one clean hit was meant.
                 Measured: two noise hits 60 or 120 ms apart read 2, a gated grain texture 1 (pro_impact short / air /
                 swell 1, deep 2, rumble 3), impact swell builds 220 ms, logo_sting zip_ping 140 ms, impact classic
                 and pro_impact short none. Without a hit role they are notes for impact / logo-sting categories.
Warnings make ok False; notes are information. Every bank render runs check() (bank.render meta['checks']).
"""
import os

import numpy as np

from ..report import emit

SR = 48000
GLIDE_ST = 4.0
GLIDE_MIN_S = 0.05
PRE_SWELL_MS = 100.0
SILENT_DBFS = -60.0
SILENT_MS = 5.0
HIT_ROLES = ('hit', 'land')
GATED = ('glitch', 'pro_glitch', 'riser', 'pro_riser', 'pro_zip', 'pro_impact')
DESIGNED = ('glitch', 'pro_glitch')
HIT_CATEGORIES = ('impact', 'pro_impact', 'logo_sting', 'pro_logo_sting')


def _mono(x):
    x = np.asarray(x, np.float64)
    return x.mean(1) if x.ndim == 2 else x


def _load(x, sr=SR):
    if isinstance(x, str):
        from .. import audio
        return audio.load(x)
    x = np.asarray(x, np.float64)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR)
    return x


# ------------------------------------------------------------------------------------------------------ detectors
def rising_glide(x, conc=0.5, jump_st=3.0, min_frames=6):
    """{'tonal_share', 'rise_st', 'from_s', 'to_s'}: the largest climb of the strongest partial (module doc)"""
    m = _mono(x)
    nfft, hop, pad = 1024, 240, 8192
    if len(m) < nfft:
        m = np.pad(m, (0, nfft - len(m)))
    k = 1 + (len(m) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(k)[:, None]
    S = np.abs(np.fft.rfft(m[idx] * np.hanning(nfft), pad, axis=1)) ** 2
    f = np.fft.rfftfreq(pad, 1 / SR)
    a, b = np.searchsorted(f, [80.0, 10000.0])
    B, fb = S[:, a:b], f[a:b]
    tot = B.sum(1) + 1e-30
    act = 10 * np.log10(tot) > 10 * np.log10(tot.max()) - 30
    fp = fb[np.argmax(B, 1)]
    half = np.maximum(fp * (2 ** (3 / 12) - 1), 2.0 * SR / nfft)        # +-3 st, at least the window's main lobe
    c = np.cumsum(B, 1)
    rows = np.arange(k)
    lo = np.clip(np.searchsorted(fb, fp - half), 0, len(fb) - 1)
    hi = np.clip(np.searchsorted(fb, fp + half), 0, len(fb) - 1)
    ton = act & ((c[rows, hi] - c[rows, lo]) / tot >= conc)
    st = 12 * np.log2(fp / 440.0)
    t = rows * hop / SR + nfft / 2 / SR
    best = [0.0, None, None]

    def close(seg):
        if len(seg) < min_frames:
            return
        v = st[seg]
        v = np.array([np.median(v[max(0, q - 2):q + 3]) for q in range(len(v))])
        rise = v - np.minimum.accumulate(v)
        q = int(np.argmax(rise))
        if rise[q] > best[0]:
            best[:] = [float(rise[q]), float(t[seg[int(np.argmin(v[:q + 1]))]]), float(t[seg[q]])]
    seg = []
    for i in np.nonzero(ton)[0]:
        if seg and (i - seg[-1] > 2 or abs(st[i] - st[seg[-1]]) > jump_st):
            close(seg)
            seg = []
        seg.append(i)
    close(seg)
    return {'tonal_share': round(float(ton.sum() / max(1, act.sum())), 2), 'rise_st': round(best[0], 1),
            'from_s': None if best[1] is None else round(best[1], 3),
            'to_s': None if best[2] is None else round(best[2], 3)}


def layers(x, peak_db=-6.0, dip_db=10.0, window_s=0.5, win=0.025, hop=0.005):
    """{'attacks': [s ...], 'count', 'pre_swell_ms'}: separate attacks and the build before the hit (module doc).
    The envelope is a 25 ms RMS every 5 ms: shorter windows read a gated grain texture (pro_impact) as attacks"""
    m = _mono(x)
    w, h = int(win * SR), int(hop * SR)
    if len(m) < w + 2 * h:
        return {'attacks': [0.0], 'count': 1, 'pre_swell_ms': 0.0}
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    st = np.arange(0, len(m) - w + 1, h)
    e = 10 * np.log10((c[st + w] - c[st]) / w + 1e-24)
    e -= e.max()
    k = len(e)
    i30 = int(np.nonzero(e >= -30)[0][0])
    i6 = int(np.nonzero(e >= peak_db)[0][0])
    pk = [i for i in range(k) if e[i] >= peak_db and e[i] >= e[max(0, i - 1)] and e[i] >= e[min(k - 1, i + 1)]]
    ev = []
    for i in pk:
        if ev and (i - ev[0]) * h / SR > window_s:
            break
        if ev and e[ev[-1]:i + 1].min() > min(e[ev[-1]], e[i]) - dip_db:
            if e[i] > e[ev[-1]]:
                ev[-1] = i
            continue
        ev.append(i)
    return {'attacks': [round((i * h + w / 2) / SR, 3) for i in ev], 'count': len(ev),
            'pre_swell_ms': round((i6 - i30) * 1000.0 * h / SR, 1)}


def live_ms(x, db=-40.0):
    """how long a sound stays within db of its loudest moment: 1 ms RMS frames every 0.25 ms, counted (ms)"""
    m = _mono(x)
    w, h = SR // 1000, SR // 4000
    if len(m) < w:
        m = np.pad(m, (0, w - len(m)))
    c = np.concatenate([[0.0], np.cumsum(m * m)])
    st = np.arange(0, len(m) - w + 1, h)
    e = (c[st + w] - c[st]) / w
    if not len(e) or e.max() <= 0:
        return 0.0
    return float((e >= e.max() * 10 ** (db / 10)).sum() * 1000.0 * h / SR)


def _bubble(glide):
    """a climb a water drop's bubble makes (earshot.forge.recipes.drip.allows): the drip category's allowance"""
    from .recipes import drip
    return drip.allows(glide)


def _ends_on_peak(x):
    m = np.abs(_mono(x))
    return len(m) > 0 and int(np.argmax(m)) >= 0.97 * len(m)


def check(x, sr=SR, category=None, role=None, sound=None):
    """the model-free audit of one sound (module doc): {'ok', 'warnings': [{code, msg}], 'notes': [...], 'summary',
    'measures', 'glide', 'layers'}. x = samples at sr (or a path); category / role / sound give context"""
    from .polish import measure
    y = _load(x, sr)
    m = measure(y)
    g = rising_glide(y)
    lay = layers(y)
    warn, notes = [], []

    def add(code, msg, soft=False):
        (notes if soft else warn).append({'code': code, 'msg': msg})
    designed = category in DESIGNED
    gated = category in GATED or _ends_on_peak(y)
    pk = float(np.abs(y).max()) if np.size(y) else 0.0
    peak_dbfs = 20 * np.log10(pk) if pk > 0 else -np.inf
    live = live_ms(y)
    if peak_dbfs < SILENT_DBFS or live < SILENT_MS:
        add('silent', 'digital silence: nothing to hear' if pk == 0 else
            f'nearly silent: peak {peak_dbfs:.0f} dBFS, {live:.1f} ms within 40 dB of its peak (a take faded to '
            'nothing, or a level far too low)')
    if m['clicks']:
        add('clicks', f"{m['clicks']} click{'s' if m['clicks'] != 1 else ''} at "
            + ', '.join(f'{t:.3f} s' for t in m['click_times'][:4]), soft=designed)
    if m['edge_db'] > -40:
        add('edge', f"starts or ends off zero ({m['edge_db']:.0f} dB re peak): it clicks when played")
    if m['clip_runs'] or m['overs']:
        add('clipping', f"clipped: {m['clip_runs']} flat tops, {m['overs']} samples over full scale")
    if m['dc_db'] > -40:
        add('dc', f"DC offset {m['dc_db']:.0f} dB re peak")
    if m['ring_z'] is not None and m['ring_z'] > 15:
        add('ring', f"metallic comb ring in the tail every {m['ring_ms']:.1f} ms (z {m['ring_z']:.0f})")
    if m['trunc_db'] is not None and m['trunc_db'] > -45:
        add('truncated', f"tail cut at {m['trunc_db']:.0f} dB re peak" + (' (the stop is the design)' if gated else
                                                                          ': fade it out or render it longer'),
            soft=gated)
    if m['silence_ms'] > 4:
        add('silence', f"{m['silence_ms']:.0f} ms of digital silence inside", soft=designed)
    if m['bump_db'] > 6:
        add('tail', f"the tail rises again by {m['bump_db']:.0f} dB (a second event, a roll or a double member)",
            soft=True)
    if g['rise_st'] > GLIDE_ST and (g['to_s'] - g['from_s']) >= GLIDE_MIN_S:
        bubble = category == 'drip' and _bubble(g)
        add('rising_glide', f"pitch climbs {g['rise_st']:.0f} semitones over {1000 * (g['to_s'] - g['from_s']):.0f} "
                            f"ms ({g['from_s']:.2f}-{g['to_s']:.2f} s)" + (
                                ': a bubble rising (drip allowance)' if bubble else
                                ': a rising synth whip reads as cheap'), soft=bubble)
    hit = role in HIT_ROLES
    if hit or category in HIT_CATEGORIES:
        if lay['count'] >= 2:
            add('stacked', f"{lay['count']} separate attacks ({', '.join(f'{t:.2f}' for t in lay['attacks'])} s): "
                           'layers stacked where one clean hit reads better', soft=not hit)
        if lay['pre_swell_ms'] > PRE_SWELL_MS:
            add('pre_swell', f"{lay['pre_swell_ms']:.0f} ms of build before the hit (a lead-in stacked onto it)",
                soft=not hit)
    keep = ('clicks', 'edge_db', 'clip_runs', 'overs', 'dc_db', 'ring_z', 'trunc_db', 'silence_ms', 'bump_db',
            'attack_ms', 't40_ms', 'crest_db', 'res_db', 'res_hz')
    extra = {'peak_dbfs': round(peak_dbfs, 2) if np.isfinite(peak_dbfs) else None, 'live_ms': round(live, 2)}
    name = sound or (os.path.basename(x) if isinstance(x, str) else 'sound')
    if warn:
        summary = f"{name}: {len(warn)} warning{'s' if len(warn) != 1 else ''}: " + '; '.join(w['msg'] for w in warn)
    else:
        summary = f'{name}: clean' + (f" ({'; '.join(n['msg'] for n in notes)})" if notes else '')
    return {'ok': not warn, 'warnings': warn, 'notes': notes, 'summary': summary,
            'measures': {**{k: m.get(k) for k in keep}, **extra}, 'glide': g, 'layers': lay,
            'context': {k: v for k, v in (('category', category), ('role', role), ('sound', sound)) if v}}


# --------------------------------------------------------------------------------------------------------- the API
def sfx_list(category=None, kind=None):
    """the bank as rows (bank.list_sounds) with a summary"""
    from . import bank
    rows = bank.list_sounds(category=category, kind=kind)
    what = ' '.join(x for x in (f'category {category}' if category else '', f'kind {kind}' if kind else '') if x)
    return {'summary': f"{len(rows)} sounds in the bank {bank.VERSION}" + (f' ({what})' if what else ''),
            'version': bank.VERSION, 'sounds': rows}


def write(path, y, sr=SR):
    import soundfile as sf
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    sf.write(path, np.asarray(y, np.float32), int(sr), subtype='PCM_24')
    return path.replace(chr(92), '/')


def sfx_render(sound, out=None, seed=None, dur=None, apex=None, pitch=None, peak_db=-1.0, weight=None, sr=SR):
    """render one take (bank.render) and write it to `out` (24-bit WAV) when given: {'summary', 'file', 'meta'}"""
    from . import bank
    y, meta = bank.render(sound, seed=seed, dur=dur, apex=apex, pitch=pitch, peak_db=peak_db, weight=weight, sr=sr)
    f = write(out, y, sr) if out else None
    return {'summary': _render_line(meta, f), 'file': f, 'meta': meta}


def _render_line(meta, f=None):
    ed = meta.get('edits') or {}
    bits = [f"{meta['id']} seed {meta['seed']}: {meta['dur_s']:.3f} s, onset {meta['onset_s']:.3f} s, peak at "
            f"{meta['peak_s']:.3f} s, end {meta['end_s']:.3f} s, {meta['peak_db']:.1f} dBFS"]
    if meta.get('lufs_m_max') is not None:
        bits[0] += f", {meta['lufs_m_max']:.1f} LUFS momentary max"
    if 'apex' in ed:
        bits.append(f"apex {ed['apex']['target']:.3f} s (measured {ed['apex']['measured']:.3f}, "
                    f"{ed['apex']['err_ms']:+.0f} ms)")
    if 'dur' in ed:
        bits.append(f"dur {ed['dur']['target']:.3f} s ({ed['dur']['method']}, {ed['dur'].get('fit', 'exact')})")
    if 'pitch' in ed:
        p = ed['pitch']
        bits.append(f"pitch {p.get('note', p.get('key'))}" + (f" (measured {p['measured_hz']:.1f} Hz)"
                                                             if p.get('measured_hz') else ''))
    c = meta.get('checks')
    if c:
        bits.append('checks: ' + ('clean' if c['ok'] else '; '.join(w['msg'] for w in c['warnings'])))
    return (f'wrote {f}: ' if f else '') + '; '.join(bits)


def sfx_check(path, category=None, role=None):
    """check() on a file (or a list of files): one result, or {'summary', 'files': [...]}"""
    if isinstance(path, (list, tuple)):
        rows = [sfx_check(p, category, role) for p in path]
        bad = sum(1 for r in rows if not r['ok'])
        return {'summary': f'{len(rows) - bad} / {len(rows)} clean', 'ok': not bad, 'files': rows}
    return check(path, category=category, role=role, sound=os.path.basename(path)) | {'file': path}


# ------------------------------------------------------------------------------------------------------------ CLI
def add_cli(sub):
    from ..cli import common
    p = sub.add_parser('sfx', help='render one sound of the shipped bank by id, aimed at the picture; audit any '
                                   'sound effect (no models)')
    s = p.add_subparsers(dest='sfx_cmd', metavar='<list|render|check>')
    pl = common(s.add_parser('list', help='the bank: ids, lengths, peak times, cue kinds, knobs'))
    pl.add_argument('--category', help='one category (whoosh_fast, pro_impact, typing_key, pluck ...)')
    pl.add_argument('--kind', help='a cue kind or kit role (whoosh, whoosh.long, hit, key, bell ...)')
    pr = common(s.add_parser('render', help='render one take to a WAV'))
    pr.add_argument('sound', help='sound id, e.g. whoosh_fast/crisp (see earshot sfx list)')
    pr.add_argument('-o', '--output', required=True, help='WAV to write (24-bit)')
    pr.add_argument('--seed', type=int, help="noise seed: a new take (default: the sound's own, the delivered final)")
    pr.add_argument('--dur', type=float, help='length in seconds (exact)')
    pr.add_argument('--apex', type=float, help='seconds from the start to the loudest moment (whoosh family)')
    pr.add_argument('--pitch', help="note name ('E5', 'Eb6') or Hz, through the recipe's own pitch parameter")
    pr.add_argument('--peak', type=float, default=-1.0, help='sample peak in dBFS, 0 or less (default -1)')
    pr.add_argument('--weight', type=float, help='level relative to --peak, 6 dB per doubling (0.5 = 6 dB down)')
    pr.add_argument('--sr', type=int, default=SR, choices=(44100, 48000, 88200, 96000),
                    help='sample rate of the WAV (default 48000)')
    pc = common(s.add_parser('check', help='model-free audit: clicks, truncation, clipping, DC, ring, tail, rising '
                                           'pitch glide, stacked hits'))
    pc.add_argument('files', nargs='+')
    pc.add_argument('--category', help='bank category, for context (a riser may stop on its peak)')
    pc.add_argument('--role', help='kit role, for context (hit / land: stacked layers and pre-swells are warnings)')

    def run(a):
        if a.sfx_cmd == 'list':
            r = sfx_list(a.category, a.kind)
            lines = [r['summary'], f"{'id':30s} {'dur s':>6s} {'peak s':>6s}  kinds / knobs"]
            for x in r['sounds']:
                kn = x['knobs']
                k = [f"dur {kn['dur']}"] + (['apex'] if kn['apex'] else []) + (
                    [f"pitch {kn['pitch'].get('note')}"] if kn['pitch'] else [])
                lines.append(f"{x['id']:30s} {x['dur_s'] or 0:6.3f} {x['peak_s'] or 0:6.3f}  "
                             f"{','.join(x['kinds']) or '-'} | {', '.join(k)}")
            emit(r, a, text='\n'.join(lines))
            return 0
        if a.sfx_cmd == 'render':
            r = sfx_render(a.sound, a.output, seed=a.seed, dur=a.dur, apex=a.apex, pitch=a.pitch, peak_db=a.peak,
                           weight=a.weight, sr=a.sr)
            emit(r, a, text=r['summary'])
            return 0
        if a.sfx_cmd == 'check':
            r = sfx_check(a.files if len(a.files) > 1 else a.files[0], a.category, a.role)
            rows = r.get('files') or [r]
            text = '\n'.join([r['summary']] * (len(rows) > 1) + [x['summary'] for x in rows])
            emit(r, a, text=text)
            return 0 if r['ok'] else 1
        p.print_help()
        return 2
    p.set_defaults(func=run)
    return p
