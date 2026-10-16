"""Gallery: one static HTML page that showcases forge results, for people who will listen to them.

    earshot gallery out/forge_v2 -o out/forge_v2/gallery.html
    earshot gallery out/forge_v2 -o gallery.html --results eval/forge/results_v2r2.json --refs forge
    earshot gallery --compare r2:out/forge_v2/*/final_r2 r3:out/forge_v2/*/final -o compare.html \\
        --results r2:eval/forge/results_v2r2.json
    earshot gallery out/forge_v2 -o showcase.html --embed-compressed        # ONE file (MP3 inside) that plays anywhere
    earshot gallery --compare "v1:demo/forge-test/*/final" "v2:out/forge_v2/*/final" --alias keypress=typing_key \\
        whoosh=whoosh_fast -o v1_v2.html                                     # v1 recipe folders on the v2 cards
    from earshot.gallery import gallery; gallery('out/forge_v2', 'gallery.html')

What goes in: a folder of forge outputs. `<folder>/<category>/final/*.wav` (+ report.json) per category, or one
category folder, or one final folder. --sub picks another subfolder (final_r2). --compare takes LABEL:GLOB pairs
(the shell must not expand them; quote them) and shows the versions as rows of one card per category, variant under
variant, so before / after reads top to bottom. The category of a folder is its name, or its parent's name when the
folder is called final* (out/forge_v2/click/final_r2 -> click); --alias OLD=NEW renames one (v1's keypress folder
on the typing_key card).

What comes out: one HTML file, no server, no network, no scripts from elsewhere (inline CSS and a short inline script:
one player at a time, and a 'match loudness' switch, on by default, that turns every player louder than the page's
10th-percentile momentary-max LUFS down to it; volume only, the files are untouched). Audio is referenced by relative
path (file:// URL across drives), or embedded as base64 with --embed (FLAC by default, lossless and about half the
size of the WAVs; --embed-codec wav keeps the original bytes). --embed-compressed makes one small self-contained page
to send or open on a phone: every sound encoded by ffmpeg (MP3 by default, 192 kbps: plays in every current browser;
--embed-codec opus / vorbis for Ogg, smaller per quality but not in older Safari; --kbps sets the rate), thumbnails as
JPEG, links to local files (report.md, compare.png, LOG.md) left out because they would be dead in a sent file, and a
note when the page passes --max-mb (25). An MP3 preview may report a few tens of ms more than the WAV (frame
padding; 183 forge sounds at 192 kbps make a ~9 MB page). Every number on the page (length, peak, LUFS, spectrogram)
is measured on the original WAVs, not on the previews; encodings are cached by content in `<html stem>_files/enc/`.
When embedding, a file shown twice (an unchanged version in --compare) goes into the page once (the script copies it).
Spectrogram thumbnails (log frequency 30 Hz-20 kHz, power averaged per pixel, 72 dB range, from the onset, one time
span per category card so lengths compare) are PNGs in `<html stem>_files/` next to the page (embedded too with
--embed); they are cached by file content and stale ones are removed.

Per category: the category's meaning, the critic's target and mode, critic passes, the held-out benchmark summary when
a results JSON is given (--results, from eval/forge/bench.py: variety, tail, tone, inside checks; made vs the forge refs
vs bench leave-one-out), the recipe and every variant's parameters (collapsible table), and links to report.md /
compare.png / LOG.md. Per sound: player, spectrogram, duration, sample peak, momentary-max loudness (LUFS, K-weighted
400 ms, as the forge reports it), critic total / pass, tone / tail numbers, benchmark numbers and flags.

Honest benchmark numbers: a results entry is used only for the file it measured. A file newer than the results'
'created' minute, or whose length / peak differ from the entry, gets none (a renamed file is matched to a unique entry
by length within 1.5 ms and peak; every case is listed in the page's notes), and a category summary is shown only
when every sound matched. With --compare, results must be labelled (r2:results.json): the same names exist in every
version. A version without its own results borrows another's numbers only for files still at the exact path that
benchmark measured and unchanged since; files identical to an earlier version's are marked '= r2'.

References: --refs forge adds an A/B row under the newest version, the nearest forge reference of each made sound (the
critic's nearest_ref, else the forge refs in order; one shared ref is shown once), plus the category's audition.wav
when its audition.txt index lists only refs that are not held out. --refs DIR reads
DIR/<category>/[forge/]*.wav. Bench references never go on a page: 'bench' is refused, and every ref path is checked
with earshot.forge.refsets.held_out() (the registry's bench files and the bench-source kit WAVs) and dropped if held
out. The benchmark numbers are shown, never the bench file names. Reference recordings are licensed third-party
sounds: keep a page that plays them private.

Returns {'html', 'assets', 'versions', 'categories', 'sounds', 'refs', 'images', 'embedded', 'codec', 'size_mb',
'warnings'}. Needs numpy, scipy, soundfile, matplotlib (image writing only; no pyplot); --embed-compressed also needs
ffmpeg on PATH (and Pillow for the JPEG thumbnails; PNG without it).
"""
import base64
import glob
import hashlib
import html
import io
import json
import math
import os
import re
import threading
import time
import urllib.parse

import numpy as np

SR = 48000
SPEC_VERSION = 6          # bump when the thumbnails change (cache key)
SPEC_W, SPEC_H = 520, 132
FLOOR_DB = 72.0
F_LO, F_HI = 30.0, 20000.0

# family -> (title, one line, categories in display order); unknown categories are placed by their base name
FAMILIES = [
    ('whoosh', 'Whooshes and swipes', 'air moved past the microphone: transitions, pass-bys, cloth and paper slides',
     ['whoosh_fast', 'whoosh_big', 'pro_whoosh_big', 'swish', 'swipe']),
    ('sweep', 'Risers and sweeps', 'builds into a cut and falls away after it',
     ['riser', 'pro_riser', 'downlifter', 'pro_zip']),
    ('impact', 'Impacts, sub drops and stings', 'the hit on the cut, the weight under it and the logo ending',
     ['impact', 'pro_impact', 'sub_drop', 'pro_sub_drop', 'logo_sting', 'pro_logo_sting']),
    ('ui', 'Interface sounds', 'clicks, taps, toggles and hover ticks for on-screen UI',
     ['click', 'pro_click', 'tap', 'pro_tap', 'pro_pop_in', 'toggle', 'hover_tick', 'pro_hover_tick']),
    ('tonal', 'Pops, dings and shimmers', 'short pitched accents: pops, notification chimes, sparkles',
     ['pop', 'pro_pop', 'notify', 'pro_notify', 'shimmer', 'pro_shimmer']),
    ('digital', 'Glitches and typing', 'digital stutters and keyboard presses',
     ['glitch', 'pro_glitch', 'typing_key', 'pro_typing_key', 'pro_typing_run']),
]
_KEYWORDS = [('whoosh', ('whoosh', 'swish', 'swipe', 'swoosh')), ('sweep', ('riser', 'downlifter', 'zip', 'sweep')),
             ('impact', ('impact', 'sub_drop', 'boom', 'sting', 'hit')),
             ('ui', ('click', 'tap', 'toggle', 'hover', 'pop_in', 'button')),
             ('tonal', ('pop', 'notify', 'ding', 'chime', 'shimmer', 'bell')),
             ('digital', ('glitch', 'typing', 'key', 'stutter'))]

PRO_DESC = {
    'pro_whoosh_big': 'the designed swells into a cut of the reference launch films',
    'pro_riser': 'the designed risers of the reference launch films',
    'pro_zip': 'a designed zip: a fast pitch sweep up in about 0.2 s',
    'pro_impact': 'the designed hits of the reference launch films',
    'pro_sub_drop': 'a designed sub-bass drop under a hit',
    'pro_logo_sting': 'the designed end sting of an effects-only launch film',
    'pro_click': 'the designed UI clicks of the reference launch films',
    'pro_tap': 'a designed tap: a 150 Hz thump with a faint click',
    'pro_pop_in': 'the most used effect of the launch films: an element popping in, a soft low thump and a faint tick',
    'pro_hover_tick': 'the designed hover ticks of the reference launch films',
    'pro_pop': 'the designed pops of the reference launch films',
    'pro_notify': 'the designed notification dings of the reference launch films',
    'pro_shimmer': 'the designed shimmers of the reference launch films',
    'pro_glitch': 'the designed glitch bursts of the reference launch films',
    'pro_typing_key': 'the designed key presses of the reference launch films',
    'pro_typing_run': 'a designed run of key presses, typing on screen',
}

# BS.1770 K-weighting at 48 kHz (same coefficients as earshot.forge.polish; copied so the gallery never imports the
# forge package, which other people may be editing while a gallery is built)
_K_SOS = np.array([[1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
                   [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]])


# ----------------------------------------------------------------------------------------------------------- audio
def _read(path):
    """(n, ch) float32 at its own rate, and the rate"""
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    return x, sr


def _to48k(x, sr):
    if sr == SR:
        return x
    from scipy import signal
    g = math.gcd(int(sr), SR)
    return signal.resample_poly(x, SR // g, int(sr) // g, axis=0).astype(np.float32)


def lufs_max(x, sr=SR):
    """momentary-max loudness in LUFS: K-weighted 400 ms windows over the channel sum of powers, zero-padded when the
    sound is shorter than 400 ms (as a meter shows it; the number the forge reports as 'momentary max LUFS')"""
    from scipy import signal
    x = _to48k(np.asarray(x, np.float32).reshape(len(x), -1), sr).astype(np.float64)
    if not len(x):
        return None
    p = (signal.sosfilt(_K_SOS, x, axis=0) ** 2).sum(axis=1)
    n = int(0.4 * SR)
    if len(p) < n:
        ms = p.sum() / n
    else:
        c = np.concatenate([[0.0], np.cumsum(p)])
        ms = (c[n:] - c[:-n]).max() / n
    return round(float(-0.691 + 10 * np.log10(ms + 1e-24)), 1)


def measure(x, sr):
    """{'dur_s', 'peak_dbfs', 'lufs', 'channels'} of a decoded sound"""
    pk = float(np.abs(x).max()) if x.size else 0.0
    return {'dur_s': round(len(x) / sr, 3), 'peak_dbfs': round(20 * math.log10(pk), 1) if pk > 0 else None,
            'lufs': lufs_max(x, sr) if x.size else None, 'channels': int(x.shape[1])}


def _onset(m, db=-60.0, pre_s=0.005, sr=SR):
    """index from which to draw: `pre_s` before the first sample within `db` of the peak (leading silence only)"""
    a = np.abs(m)
    if not len(a) or a.max() <= 0:
        return 0
    on = int(np.argmax(a > a.max() * 10 ** (db / 20)))
    return max(0, on - int(pre_s * sr))


def _cmap():
    """white paper to blue-black ink: quiet bins disappear into the page, loud ones read as ink"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('earshot_ink', [
        (0.00, '#fbfbfa'), (0.18, '#e6eaef'), (0.42, '#a9b8c9'), (0.62, '#5f7793'), (0.80, '#283b55'),
        (1.00, '#070c14')])


def _band_mean(P, centers, edges, axis_vals):
    """rows of P (bins along axis 0 at axis_vals) averaged inside each [edges[k], edges[k+1]); a band narrower than
    one bin takes the linear interpolation at its centre"""
    C = np.concatenate([np.zeros((1,) + P.shape[1:]), np.cumsum(P, axis=0)])
    lo = np.searchsorted(axis_vals, edges[:-1])
    hi = np.searchsorted(axis_vals, edges[1:])
    cnt = (hi - lo).astype(float)
    out = np.empty((len(centers),) + P.shape[1:])
    has = cnt > 0
    out[has] = (C[hi[has]] - C[lo[has]]) / cnt[has][:, None]
    if (~has).any():
        idx = np.interp(centers[~has], axis_vals, np.arange(len(axis_vals)))
        i0 = np.clip(np.floor(idx).astype(int), 0, len(axis_vals) - 1)
        i1 = np.clip(i0 + 1, 0, len(axis_vals) - 1)
        fr = (idx - i0)[:, None]
        out[~has] = P[i0] * (1 - fr) + P[i1] * fr
    return out


def spectrogram_rgb(x, sr, span_s, w=SPEC_W, h=SPEC_H):
    """(h, w, 3) uint8 log-frequency spectrogram of a sound over [onset, onset + span_s), 0 dB = its loudest cell.
    Power is averaged inside each pixel's frequency band and time slot (no speckle from point sampling)."""
    from scipy import signal
    m = np.asarray(x, np.float64)
    m = m.mean(axis=1) if m.ndim == 2 else m
    m = m[_onset(m, sr=sr):]
    n_span = max(int(span_s * sr), 64)
    m = np.pad(m[:n_span], (0, max(0, n_span - len(m))))
    nfft = 256 if span_s < 0.15 else 512 if span_s < 0.6 else 1024 if span_s < 2.5 else 2048
    nfft = min(nfft, max(64, 2 ** int(np.log2(len(m)))))
    hop = int(np.clip(n_span // (2 * w), 8, nfft // 2))
    f, t, Z = signal.stft(m, sr, nperseg=nfft, noverlap=nfft - hop, boundary='zeros', padded=True)
    P = np.abs(Z) ** 2
    # frequency: log-spaced pixel rows, bottom = F_LO
    edges = np.geomspace(F_LO, min(F_HI, sr / 2), h + 1)
    fy = np.sqrt(edges[:-1] * edges[1:])
    P = _band_mean(P, fy, edges, f)
    # time: pixel columns over the span (frames centred at t)
    te = np.linspace(0, n_span / sr, w + 1)
    P = _band_mean(P.T, (te[:-1] + te[1:]) / 2, te, t).T
    # rows narrower than a few FFT bins see one noisy bin per frame (vertical streaks): average them over time until
    # each pixel holds about 4 independent cells (at most w/16 columns, a triangular kernel)
    dt_col = n_span / sr / w
    nb = np.maximum(np.diff(edges) / (sr / nfft), 1.0)
    for k in np.nonzero(nb < 4)[0]:
        L = int(min(w // 16, round(4 * (nfft / sr) / nb[k] / dt_col)))
        if L >= 3:
            ker = np.bartlett(L + 2)[1:-1]
            P[k] = np.convolve(P[k], ker / ker.sum(), mode='same')
    L = 10 * np.log10(P + 1e-20)
    L = np.maximum(L - L.max(), -FLOOR_DB)
    v = (L[::-1] + FLOOR_DB) / FLOOR_DB
    rgb = _cmap()(np.clip(v, 0, 1))[..., :3]
    for g in (100.0, 1000.0, 10000.0):        # faint frequency guides, drawn only over quiet cells
        if fy[0] < g < fy[-1]:
            r = h - 1 - int(np.argmin(np.abs(fy - g)))
            quiet = v[r] < 0.35
            rgb[r, quiet] = rgb[r, quiet] * 0.9 + np.array([0.80, 0.83, 0.87]) * 0.1 - 0.035
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def _save_png(path, rgb):
    from matplotlib import image
    image.imsave(path, rgb)


# ------------------------------------------------------------------------------------------------------- discovery
def _norm(p):
    return os.path.abspath(p).replace(chr(92), '/')


def _wavs(d):
    """sound files of a final folder (the audition reel excluded), sorted"""
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f).replace(chr(92), '/') for f in os.listdir(d)
                  if f.lower().endswith(('.wav', '.flac')) and not f.lower().lstrip('_').startswith('audition'))


def category_of(d):
    """out/forge_v2/click/final_r2 -> click; out/forge_v2/click -> click"""
    d = os.path.normpath(os.path.abspath(d))
    base = os.path.basename(d)
    return os.path.basename(os.path.dirname(d)) if base.lower().startswith('final') else base


def discover(folder, sub='final'):
    """{category: final dir} for a forge output tree, one category folder or one final folder"""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'no such folder: {folder}')
    if _wavs(folder):
        return {category_of(folder): _norm(folder)}
    if _wavs(os.path.join(folder, sub)):
        return {os.path.basename(folder): _norm(os.path.join(folder, sub))}
    out = {}
    for c in sorted(os.listdir(folder)):
        d = os.path.join(folder, c, sub)
        if not c.startswith(('_', '.')) and _wavs(d):
            out[c] = _norm(d)
    if not out:
        raise FileNotFoundError(f'no forge results under {folder} (looked for <category>/{sub}/*.wav)')
    return out


def _split_label(s):
    """'r2:out/x' -> ('r2', 'out/x'); 'D:/out/x' and 'out/x' -> (None, s) (a one-letter prefix is a drive)"""
    m = re.match(r'^([A-Za-z0-9_.+-]{2,}):(.+)$', s)
    return (m.group(1), m.group(2)) if m else (None, s)


def parse_alias(alias):
    """{'keypress': 'typing_key'} from {'keypress': 'typing_key'} or ['keypress=typing_key', ...] (None -> {})"""
    if not alias:
        return {}
    if isinstance(alias, dict):
        return {str(k): str(v) for k, v in alias.items()}
    out = {}
    for a in alias:
        if '=' not in a or not all(q.strip() for q in a.split('=', 1)):
            raise ValueError(f'--alias needs OLD=NEW pairs (e.g. keypress=typing_key), got {a!r}')
        k, v = a.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def compare_versions(specs, alias=None):
    """[('r2', {cat: dir}), ('r3', {cat: dir})] from ['r2:GLOB', 'r3:GLOB', ...] (one label may repeat); alias renames
    folder categories ({'keypress': 'typing_key'}: an older version's folder shown on the newer category's card)"""
    alias = parse_alias(alias)
    out = {}
    for s in specs:
        label, pat = _split_label(s)
        if not label:
            raise ValueError(f'--compare needs LABEL:GLOB pairs (e.g. r2:out/forge_v2/*/final_r2), got {s!r}')
        hits = sorted(glob.glob(pat)) or ([pat] if os.path.isdir(pat) else [])
        cats = out.setdefault(label, {})
        for d in hits:
            if not os.path.isdir(d):
                continue
            if not _wavs(d) and _wavs(os.path.join(d, 'final')):
                d = os.path.join(d, 'final')
            if _wavs(d) and not category_of(d).startswith(('_', '.')):    # _judge, _tools: not categories
                c = category_of(d)
                cats.setdefault(alias.get(c, c), _norm(d))
    if not out or not any(out.values()):
        raise FileNotFoundError(f'--compare found no folders with sounds: {specs}')
    return list(out.items())


def family_of(cat):
    for key, _, _, cats in FAMILIES:
        if cat in cats:
            return key
    base = cat[4:] if cat.startswith('pro_') else cat
    for key, _, _, cats in FAMILIES:
        if base in cats:
            return key
    for key, words in _KEYWORDS:
        if any(w in cat for w in words):
            return key
    return 'other'


def _cat_order(cat):
    for fi, (_, _, _, cats) in enumerate(FAMILIES):
        if cat in cats:
            return fi, cats.index(cat), cat
    return len(FAMILIES), 99, cat


def describe_category(cat):
    try:
        from .forge.refsets import CATEGORIES   # plain dict module-level constant
    except Exception:
        CATEGORIES = {}
    if cat in PRO_DESC:
        return PRO_DESC[cat]
    if cat in CATEGORIES:
        return CATEGORIES[cat]
    base = cat[4:] if cat.startswith('pro_') else cat
    if base in CATEGORIES:
        return 'designed version: ' + CATEGORIES[base]
    return ''


def humanize(cat):
    base = cat[4:] if cat.startswith('pro_') else cat
    return base.replace('_', ' ').capitalize()


# --------------------------------------------------------------------------------------------------------- reports
def _load_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def _variant(p):
    for k in ('member', 'family', 'preset', 'style', 'variant'):
        v = p.get(k)
        if isinstance(v, str) and v:
            return v
    return ''


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def critic_of(pick):
    """the critic's numbers for one pick, whichever report layout wrote it"""
    s = pick.get('score') or {}
    me = pick.get('measured') or {}
    rob = pick.get('robust') or {}
    pol = me.get('polish') if isinstance(me.get('polish'), dict) else {}
    words = s.get('polish_words') or pol.get('words') or ''
    hold = pick.get('holdout') or {}
    top = hold.get('audioset_top') or []
    return {
        'total': _num(s.get('total')), 'passes': s.get('passes') if isinstance(s.get('passes'), bool) else None,
        'match_z': _num(s.get('match_z')), 'tone_db': _num(s.get('tone_db', me.get('tone_db'))),
        'env_pen': _num(s.get('env_pen')), 'polish_pen': _num(s.get('polish_pen')), 'art_pen': _num(s.get('art_pen')),
        't40_ms': _num(me.get('t40_ms')), 't40_ref': me.get('t40_ref'),
        'nearest_ref': s.get('nearest_ref') or me.get('nearest_ref'),
        'sounds_like': [w for w in (s.get('sounds_like') or []) if isinstance(w, str)][:3],
        'words': words if isinstance(words, str) else '',
        'robust_mean': _num(rob.get('mean')), 'robust_pass': _num(rob.get('pass_rate')),
        'audioset': top[0][0] if top and isinstance(top[0], (list, tuple)) else None,
    }


def read_report(d):
    """(meta, {basename: pick}) from <d>/report.json ({} when it is missing or unreadable)"""
    r = _load_json(os.path.join(d, 'report.json'))
    if not isinstance(r, dict):
        return {}, {}
    picks = {}
    for p in r.get('picks') or []:
        if not isinstance(p, dict):
            continue
        name = os.path.basename(str(p.get('file') or '')) or (str(p.get('name', '')) + '.wav')
        picks[name] = p
    crit = r.get('critic') if isinstance(r.get('critic'), dict) else {}
    recipe = r.get('recipe') or r.get('recipe_module') or ''
    if r.get('recipe_module') and r['recipe_module'] != recipe:     # 'logo_sting' made by the sting module
        recipe = r['recipe_module']
    if r.get('kind') and r.get('kind') != recipe:
        recipe = f"{recipe} ({r['kind']})"
    meta = {'target': r.get('target') or '', 'mode': r.get('mode') or crit.get('mode') or '',
            'recipe': recipe, 'method': r.get('method') or r.get('summary') or '', 'refs': r.get('refs') or [],
            'tag': r.get('tag') or ''}
    return meta, picks


def _bench_index(results):
    """{category: {'made': {basename: entry (+ 'inside_n')}, 'summary', 'inside'}} from an eval/forge/bench.py JSON"""
    out = {}
    for cat, c in ((results or {}).get('categories') or {}).items():
        if not isinstance(c, dict):
            continue
        ins = c.get('inside') or {}
        per = ins.get('per_sound') or []
        made = {}
        for i, e in enumerate(c.get('made') or []):
            if isinstance(e, dict) and e.get('file'):
                made[os.path.basename(e['file'])] = dict(e, inside_n=per[i] if i < len(per) else None)
        out[cat] = {'made': made, 'summary': c.get('summary') or {}, 'inside': ins, 'n_bench': c.get('n_bench'),
                    'n_forge': c.get('n_forge')}
    return out


def bench_of(entry, n_checks):
    if not entry:
        return None
    arts = entry.get('artifacts') or []
    top = entry.get('as_top') or []
    return {'tone_db': _num(entry.get('tone_db')), 't40_ratio': _num(entry.get('t40_ratio')),
            'typ_z': _num(entry.get('typ_z')), 'phrase_z': _num(entry.get('phrase_z')),
            'inside': entry.get('inside_n'), 'n_checks': n_checks, 'stereo_corr': _num(entry.get('stereo_corr')),
            'artifacts': [a if isinstance(a, str) else str(a) for a in arts],
            'audioset': str(top[0][0]).split(',')[0] if top and isinstance(top[0], (list, tuple)) else None}


def bench_stale(entry, sound, created, len_tol=0.011):
    """why a results entry does not describe this file ('' = it does): the file changed after the benchmark ran
    (mtime later than the results' 'created' minute) or its length / peak differ from what the benchmark measured"""
    if not entry:
        return ''
    if created:
        try:
            t = time.mktime(time.strptime(created, '%Y-%m-%d %H:%M'))
            if os.path.getmtime(sound['path']) > t + 60:
                return 'file newer than the benchmark run'
        except (ValueError, OSError):
            pass
    ln, pk = _num(entry.get('length_s')), _num((entry.get('artifact_detail') or {}).get('peak_dbfs'))
    if ln is not None and abs(ln - sound['dur_s']) > len_tol:
        return f"length {sound['dur_s']:.3f} s, benchmark {ln:.3f} s"
    if pk is not None and sound.get('peak_dbfs') is not None and abs(pk - sound['peak_dbfs']) > 0.3:
        return f"peak {sound['peak_dbfs']:.1f} dBFS, benchmark {pk:.1f}"
    return ''


def _same_file(rel, path, results_file):
    """is a results JSON's made-file entry (often repo-relative) this path? tried from the current folder and from
    the repo root two levels above the results file (eval/forge/results_*.json)"""
    want = os.path.normcase(os.path.abspath(path))
    if os.path.isabs(rel):
        return os.path.normcase(os.path.abspath(rel)) == want
    roots = [os.getcwd(), os.path.join(os.path.dirname(os.path.abspath(results_file)), '..', '..')]
    return any(os.path.normcase(os.path.abspath(os.path.join(r, rel))) == want for r in roots)


def bench_summary(b):
    """category-level benchmark rows: {'rows': [(label, made, real_cross, real_loo, hint)], 'inside': ...}"""
    if not b:
        return None
    s = b.get('summary') or {}
    m, x, lo = s.get('made') or {}, s.get('real_cross') or {}, s.get('real_loo') or {}
    rows = []
    for key, label, hint, fmt in (('var_cos', 'variety', 'CLAP cosine between variants; lower = more varied', '{:.3f}'),
                                  ('t40_ratio', 'tail t40', 'tail length / bench median; 1 = the same', '{:.2f} x'),
                                  ('tone_db', 'tone distance', '1/3-octave distance to the bench median; lower = '
                                   'closer', '{:.1f} dB')):
        vals = [d.get(key) for d in (m, x, lo)]
        if any(_num(v) is not None for v in vals):
            rows.append((label, *[fmt.format(v) if _num(v) is not None else '-' for v in vals], hint))
    ins = b.get('inside') or {}
    per, real, n = ins.get('per_sound') or [], ins.get('real_per_sound') or [], ins.get('n_checks')
    if per and n:
        rows.append(('inside', f'{np.median(per):g} / {n}', f'{np.median(real):g} / {n}' if real else '-', '-',
                     'checks inside the range of real recordings (median per sound)'))
    return {'rows': rows, 'n_bench': b.get('n_bench'), 'n_forge': b.get('n_forge'), 'n_made': m.get('n'),
            'artifacts': m.get('artifacts')}


# ------------------------------------------------------------------------------------------------------ references
def refs_for(cat, refs, meta, warnings):
    """forge-side reference paths for a category (held-out ones dropped), or []"""
    if not refs:
        return []
    if refs == 'bench':
        raise ValueError("bench references are held out: they never go on a page (use --refs forge)")
    paths = []
    if refs == 'forge':
        try:
            from .forge import refsets
            paths = refsets.refset(cat, 'forge')
        except Exception:
            paths = [p for p in meta.get('refs') or [] if os.path.exists(p)]
    elif os.path.isdir(refs):
        for d in (os.path.join(refs, cat, 'forge'), os.path.join(refs, cat)):
            if _wavs(d):
                paths = _wavs(d)
                break
    else:
        raise ValueError(f"--refs must be 'forge' or a folder of <category>/ subfolders, not {refs!r}")
    if not paths:
        return []
    try:
        from .forge.refsets import held_out
        bad = set(held_out(paths))
    except Exception as e:  # without the guard no refs at all: a page must never risk a held-out file
        warnings.append(f'{cat}: references skipped, the held-out check is unavailable ({type(e).__name__}: {e})')
        return []
    if bad:
        warnings.append(f'{cat}: {len(bad)} held-out reference(s) dropped')
    return [_norm(p) for p in paths if p not in bad]


def audition_ok(wav, cat, warnings):
    """may this audition reel go on the page? only when its .txt index lists every ref it plays and none of them is
    held out (a registry bench file name or a bench-source kit stem)"""
    txt = os.path.splitext(wav)[0] + '.txt'
    try:
        with open(txt, encoding='utf-8') as fh:
            names = re.findall(r'^\s*[0-9.]+\s*s\s+ref\s+(\S+)', fh.read(), re.M)
    except OSError:
        warnings.append(f'{cat}: audition reel left out (no {os.path.basename(txt)} to check its refs against the '
                        f'held-out list)')
        return False
    try:
        from .forge import refsets
        try:
            bench = {os.path.basename(q) for c in refsets.load().get('categories', {}).values()
                     for q in c.get('bench', [])}
        except refsets.RefsetsMissing:
            bench = set()
        kit = set(refsets.KIT_BENCH)
    except Exception as e:
        warnings.append(f'{cat}: audition reel left out (held-out check unavailable: {type(e).__name__})')
        return False
    bad = [n for n in names if n in bench or os.path.splitext(n)[0] in kit]
    if bad:
        warnings.append(f'{cat}: audition reel left out (it plays held-out refs: {", ".join(bad[:3])})')
        return False
    return True


def pair_refs(sounds, ref_paths):
    """one reference per made sound: the critic's nearest_ref when it is among ref_paths, else the refs in order"""
    by_name = {os.path.basename(p): p for p in ref_paths}
    out, k = [], 0
    for s in sounds:
        near = (s.get('critic') or {}).get('nearest_ref')
        if near and near in by_name:
            out.append(by_name[near])
        elif ref_paths:
            out.append(ref_paths[k % len(ref_paths)])
            k += 1
        else:
            out.append(None)
    return out


# -------------------------------------------------------------------------------------------------------- building
# lossy codecs for --embed-compressed: ffmpeg encoder, container format, MIME type, file extension, default kbps
LOSSY = {'mp3': ('libmp3lame', 'mp3', 'audio/mpeg', '.mp3', 192),
         'opus': ('libopus', 'ogg', 'audio/ogg', '.opus', 128),
         'vorbis': ('libvorbis', 'ogg', 'audio/ogg', '.ogg', 160)}
CODECS = ('flac', 'wav') + tuple(LOSSY)


def encode_lossy(path, codec='mp3', kbps=None):
    """the bytes of `path` encoded by ffmpeg to MP3 / Ogg Opus / Ogg Vorbis at `kbps` (default per codec); 48 kHz
    (Opus' own rate; MP3 and Vorbis keep 48 kHz too), the file's own channels, no metadata"""
    import shutil
    import subprocess
    if codec not in LOSSY:
        raise ValueError(f'codec must be one of {", ".join(LOSSY)}, not {codec!r}')
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('--embed-compressed needs ffmpeg on PATH (or use --embed: lossless FLAC, larger)')
    enc, fmt, _, _, k0 = LOSSY[codec]
    cmd = [exe, '-nostdin', '-v', 'error', '-i', os.path.abspath(path), '-map_metadata', '-1', '-vn', '-ar', str(SR),
           '-c:a', enc, '-b:a', f'{int(kbps or k0)}k']
    if codec == 'mp3':
        cmd += ['-write_xing', '1']            # the LAME header carries the encoder delay, so players trim it
    cmd += ['-f', fmt, '-']
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode or not r.stdout:
        raise RuntimeError(f'ffmpeg could not encode {os.path.basename(path)} to {codec}: '
                           + r.stderr.decode('utf-8', 'replace').strip()[-300:])
    return r.stdout


def _jpeg(png_bytes, quality=90):
    """a PNG thumbnail as JPEG bytes (smooth spectrograms compress ~2-3x), or None without Pillow"""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=quality, optimize=True, progressive=True, subsampling=0)
        return buf.getvalue()
    except Exception:
        return None


class _Assets:
    """thumbnails next to the page (cached by content), media as relative URLs or data URIs"""

    def __init__(self, html_path, embed=False, codec='flac', kbps=None):
        self.html_dir = os.path.dirname(os.path.abspath(html_path))
        stem = os.path.splitext(os.path.basename(html_path))[0]
        self.dir = os.path.join(self.html_dir, f'{stem}_files')
        self.embed, self.codec = embed, codec
        self.lossy = bool(embed) and codec in LOSSY
        self.kbps = int(kbps or LOSSY[codec][4]) if self.lossy else None
        self.used, self.made, self.reused, self.embedded_bytes = set(), 0, 0, 0
        self.audio_bytes, self.image_bytes, self.encoded, self.enc_pruned = 0, 0, 0, 0
        self._decoded = {}
        self._lock = threading.Lock()
        self._audio_ids, self.deduped = {}, 0

    def url(self, path):
        if self.embed:
            return self._data_uri(path)
        try:
            rel = os.path.relpath(os.path.abspath(path), self.html_dir).replace(chr(92), '/')
        except ValueError:          # another drive: an absolute file URL still works offline
            from pathlib import Path
            return Path(os.path.abspath(path)).as_uri()
        return urllib.parse.quote(rel, safe='/._-~')

    def audio_src(self, path):
        """the src attribute(s) of an <audio> for a sound: with embedding, an identical file (same bytes, e.g. a version
        that did not change) is put in the page once and later players carry data-src-of (the inline script copies the
        first player's source), so an unchanged round does not double the page"""
        if not self.embed:
            return f' src="{html.escape(self.url(path), quote=True)}"'
        with open(path, 'rb') as fh:
            sha = hashlib.sha1(fh.read()).hexdigest()
        first = self._audio_ids.get(sha)
        if first:
            self.deduped += 1
            return f' data-src-of="{first}"'
        aid = f'a{len(self._audio_ids) + 1}'
        self._audio_ids[sha] = aid
        return f' id="{aid}" src="{html.escape(self.url(path), quote=True)}"'

    def link(self, path):
        """a plain relative link (never embedded): report.md, compare.png, LOG.md"""
        try:
            rel = os.path.relpath(os.path.abspath(path), self.html_dir).replace(chr(92), '/')
        except ValueError:
            from pathlib import Path
            return Path(os.path.abspath(path)).as_uri()
        return urllib.parse.quote(rel, safe='/._-~')

    def _data_uri(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.png':
            with open(path, 'rb') as fh:
                mime, data = 'image/png', fh.read()
            jpg = _jpeg(data) if self.lossy else None
            if jpg and len(jpg) < len(data):
                mime, data = 'image/jpeg', jpg
            self.image_bytes += len(data)
        elif ext in ('.wav', '.flac') and self.lossy:
            mime, data = LOSSY[self.codec][2], self.encoded_bytes(path)
            self.audio_bytes += len(data)
        elif ext in ('.wav', '.flac') and self.codec == 'flac':
            import soundfile as sf
            x, sr = self.decode(path)
            buf = io.BytesIO()
            sf.write(buf, x, sr, format='FLAC', subtype='PCM_24')
            mime, data = 'audio/flac', buf.getvalue()
        else:
            mime = {'.wav': 'audio/wav', '.flac': 'audio/flac'}.get(ext, 'application/octet-stream')
            with open(path, 'rb') as fh:
                data = fh.read()
        if mime.startswith('audio/') and not self.lossy:
            self.audio_bytes += len(data)
        self.embedded_bytes += len(data)
        return f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')

    def _enc_path(self, path):
        with open(path, 'rb') as fh:
            sha = hashlib.sha1(fh.read()).hexdigest()
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '-', os.path.splitext(os.path.basename(path))[0])
        return os.path.join(self.dir, 'enc', f'{stem}_{sha[:10]}_{self.kbps}k{LOSSY[self.codec][3]}')

    def encoded_bytes(self, path):
        """the lossy preview of a sound, from the content cache (<stem>_files/enc/) or ffmpeg"""
        out = self._enc_path(path)
        self.used.add(os.path.normcase(os.path.abspath(out)))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            with open(out, 'rb') as fh:
                return fh.read()
        data = encode_lossy(path, self.codec, self.kbps)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = f'{out}.{threading.get_ident()}.part'
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, out)
        with self._lock:
            self.encoded += 1
        return data

    def prefetch(self, paths, workers=3):
        """encode the previews of many sounds in a few threads (ffmpeg processes) before the page is written"""
        if not self.lossy:
            return
        todo = list(dict.fromkeys(p for p in paths if p and not os.path.exists(self._enc_path(p))))
        if todo:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                list(ex.map(self.encoded_bytes, todo))

    def decode(self, path):
        key = _norm(path)
        if key not in self._decoded:
            self._decoded[key] = _read(path)
        return self._decoded[key]

    def spectrogram(self, path, span_s, label=''):
        st = os.stat(path)
        key = f'{_norm(path)}|{st.st_size}|{st.st_mtime_ns}|{span_s:.4f}|{SPEC_VERSION}'
        h = hashlib.sha1(key.encode()).hexdigest()
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '-', f'{label}_{os.path.splitext(os.path.basename(path))[0]}').strip('-')
        out = os.path.join(self.dir, f'{stem}_{h[:8]}.png')
        self.used.add(os.path.normcase(os.path.abspath(out)))
        if os.path.exists(out):
            self.reused += 1
            return out
        os.makedirs(self.dir, exist_ok=True)
        x, sr = self.decode(path)
        _save_png(out, spectrogram_rgb(x, sr, span_s))
        self.made += 1
        return out

    def prune(self):
        """remove thumbnails this page no longer uses (only PNGs inside the page's own _files folder; with lossy
        embedding also stale previews in _files/enc/); returns the number of thumbnails removed"""
        n = 0
        if os.path.isdir(self.dir):
            for f in os.listdir(self.dir):
                p = os.path.join(self.dir, f)
                if f.endswith('.png') and os.path.normcase(os.path.abspath(p)) not in self.used:
                    try:
                        os.remove(p)
                        n += 1
                    except OSError:
                        pass
        enc = os.path.join(self.dir, 'enc')
        if self.lossy and os.path.isdir(enc):
            exts = tuple(v[3] for v in LOSSY.values()) + ('.part',)
            for f in os.listdir(enc):
                p = os.path.join(enc, f)
                if f.endswith(exts) and os.path.normcase(os.path.abspath(p)) not in self.used:
                    try:
                        os.remove(p)
                        self.enc_pruned += 1
                    except OSError:
                        pass
        return n


def _sound(path, assets, warnings, pick=None):
    try:
        x, sr = assets.decode(path)
        with open(path, 'rb') as fh:
            sha = hashlib.sha1(fh.read()).hexdigest()
    except Exception as e:
        warnings.append(f'{os.path.basename(path)}: unreadable ({type(e).__name__}: {e})')
        return None
    s = {'path': path, 'file': os.path.basename(path), 'name': os.path.splitext(os.path.basename(path))[0],
         'sha1': sha, **measure(x, sr)}
    if pick is not None:
        s['variant'] = _variant(pick)
        s['critic'] = critic_of(pick)
        s['params'] = pick.get('params') if isinstance(pick.get('params'), dict) else {}
        s['seed'] = pick.get('seed')
    else:
        s['variant'], s['critic'], s['params'], s['seed'] = '', {}, {}, None
    s['bench'] = None
    return s


def align_variants(versions):
    """display order per version (v['order'], None = an empty slot): a later version puts a sound under the earlier
    version's sound of the same variant name (crisp under crisp) when at least two names are shared; otherwise, and
    for the first version, file order"""
    base = None
    for v in versions:
        v['order'] = list(v['sounds'])
        names = [q.get('variant') for q in v['sounds']]
        if base is None:
            if v['sounds']:
                base = [q.get('variant') for q in v['sounds']]
            continue
        if not all(names) or len(set(names)) != len(names) or len(set(names) & set(base)) < 2:
            continue
        slots = [None] * max(len(base), len(names))
        rest = []
        for q in v['sounds']:
            if q['variant'] in base and slots[base.index(q['variant'])] is None:
                slots[base.index(q['variant'])] = q
            else:
                rest.append(q)
        for q in rest:
            if None in slots:
                slots[slots.index(None)] = q
            else:
                slots.append(q)
        while slots and slots[-1] is None:
            slots.pop()
        v['order'] = slots


def _span(durs):
    """one time span per card: the longest sound, but at most 2.5 x the median so one outlier cannot squash the rest"""
    d = [v for v in durs if v]
    if not d:
        return 1.0
    return float(max(0.05, min(max(d), 2.5 * float(np.median(d)))))


def build(versions, html_path, results=None, refs=None, embed=False, codec='flac', title=None, max_refs=6,
          kbps=None):
    """collect everything the page shows; returns (page dict, assets, warnings)"""
    warnings = []
    assets = _Assets(html_path, embed, codec, kbps)
    labels = [lab for lab, _ in versions]
    multi = len(versions) > 1
    # results: {label or None: bench index}
    bench = {}
    for spec in results or []:
        label, path = _split_label(spec)
        if label and label not in labels:
            raise ValueError(f'--results label {label!r} matches no version ({", ".join(labels)})')
        if not label and multi:
            raise ValueError('with --compare, label each results JSON (e.g. --results '
                             'r2:eval/forge/results_v2r2.json): the same file names exist in every version')
        r = _load_json(path)
        if not isinstance(r, dict) or 'categories' not in r:
            raise ValueError(f'not a benchmark results JSON (eval/forge/bench.py): {path}')
        bench[label or labels[0]] = {'index': _bench_index(r), 'tag': r.get('tag'), 'created': r.get('created'),
                                     'file': path}
    cats = sorted({c for _, m in versions for c in m}, key=_cat_order)
    page = {'title': title, 'labels': labels, 'multi': multi, 'families': [], 'bench': bench, 'refs': refs,
            'n_sounds': 0, 'n_refs': 0, 'n_pass': {lab: [0, 0] for lab in labels}}
    fam_map = {}
    for cat in cats:
        card = {'cat': cat, 'family': family_of(cat), 'desc': describe_category(cat), 'versions': [], 'refs': [],
                'audition': None}
        for lab, m in versions:
            d = m.get(cat)
            if not d:
                card['versions'].append({'label': lab, 'missing': True, 'sounds': []})
                continue
            meta, picks = read_report(d)
            bi = (bench.get(lab) or {}).get('index', {}).get(cat)
            n_checks = ((bi or {}).get('inside') or {}).get('n_checks')
            sounds, stale, renamed = [], [], []
            created = (bench.get(lab) or {}).get('created')
            for p in _wavs(d):
                s = _sound(p, assets, warnings, picks.get(os.path.basename(p)))
                if not s:
                    continue
                entry = (bi or {}).get('made', {}).get(os.path.basename(p)) if bi else None
                why = bench_stale(entry, s, created)
                if why and not why.startswith('file newer'):   # renamed / renumbered after the run?
                    alt = [e for e in (bi or {}).get('made', {}).values() if not bench_stale(e, s, created, 0.0015)]
                    if len(alt) == 1:
                        entry, why = alt[0], ''
                        renamed.append(f"{s['file']} = {os.path.basename(alt[0]['file'])}")
                if why:
                    stale.append(f"{s['file']} ({why})")
                s['bench'] = bench_of(entry, n_checks) if entry and not why else None
                sounds.append(s)
            if renamed:
                warnings.append(f'{cat} {lab}: {len(renamed)} file(s) were renamed after the benchmark ran, matched by '
                                f'length and peak: ' + '; '.join(renamed))
            if stale:
                more = ' ...' if len(stale) > 3 else ''
                warnings.append(f'{cat} {lab}: benchmark numbers not shown for {len(stale)} sound(s) that changed '
                                f'after the benchmark ran: ' + '; '.join(stale[:3]) + more)
            bench_from = lab if bi else None
            if not bi and sounds:           # files still at the path another version's benchmark measured, unchanged
                for ol, ob in bench.items():
                    obi = ob['index'].get(cat)
                    if not obi:
                        continue
                    hits = 0
                    for snd in sounds:
                        e = obi['made'].get(snd['file'])
                        same = e and _same_file(e['file'], snd['path'], ob['file'])
                        if same and not bench_stale(e, snd, ob.get('created'), 0.0015):
                            snd['bench'] = bench_of(e, (obi.get('inside') or {}).get('n_checks'))
                            snd['same_as'] = ol
                            hits += 1
                    if hits == len(sounds):
                        bi, bench_from = obi, ol
                        break
            matched = sum(1 for s in sounds if s['bench'])
            v = {'label': lab, 'dir': d, 'meta': meta, 'sounds': sounds,
                 'bench': bench_summary(bi) if bi and matched == len(sounds) and sounds else None,
                 'bench_from': bench_from,
                 'bench_stale': bool(bi) and matched < len(sounds),
                 'links': {} if embed else {      # a sent single file cannot reach the files next to it
                     k: q for k, q in (('report', os.path.join(d, 'report.md')),
                                       ('compare sheet', os.path.join(d, 'compare.png')),
                                       ('log', os.path.join(os.path.dirname(d), 'LOG.md'))) if os.path.exists(q)}}
            page['n_sounds'] += len(sounds)
            crit = [s['critic'].get('passes') for s in sounds]
            page['n_pass'][lab][0] += sum(1 for c in crit if c)
            page['n_pass'][lab][1] += len(sounds)
            card['versions'].append(v)
        seen = {}
        for v in card['versions']:            # a file identical to an earlier version's is marked, not hidden
            for snd in v['sounds']:
                if snd['sha1'] in seen:
                    snd['same_as'] = seen[snd['sha1']]
            for snd in v['sounds']:
                seen.setdefault(snd['sha1'], v['label'])
        align_variants(card['versions'])
        newest = next((v for v in reversed(card['versions']) if not v.get('missing')), None)
        if refs and newest:
            rp = refs_for(cat, refs, newest['meta'], warnings)
            shown = [q for q in newest['order'] if q]
            pairs = iter(pair_refs(shown, rp) if rp else [])
            pairs = [next(pairs) if q else None for q in newest['order']][:max_refs] if rp else []
            if len({q for q in pairs if q}) == 1:      # one ref for every sound (a one-ref category): show it once
                pairs = [next(q for q in pairs if q)]
            card['refs'] = [_sound(p, assets, warnings) if p else None for p in pairs]
            page['n_refs'] += sum(1 for r in card['refs'] if r)
            aud = next((q for q in (os.path.join(newest['dir'], n) for n in ('_audition.wav', 'audition.wav'))
                        if os.path.exists(q)), None)
            card['audition'] = aud if aud and audition_ok(aud, cat, warnings) else None
        allsounds = [(v['label'], s) for v in card['versions'] for s in v['sounds']] + \
            [('ref', r) for r in card['refs'] if r]
        card['span'] = _span([s['dur_s'] for _, s in allsounds])
        for lab, s in allsounds:
            try:
                s['png'] = assets.spectrogram(s['path'], card['span'], f'{lab}_{cat}')
            except Exception as e:
                warnings.append(f"{s['file']}: no spectrogram ({type(e).__name__}: {e})")
                s['png'] = None
        fam_map.setdefault(card['family'], []).append(card)
    for key, ftitle, line, _ in FAMILIES + [('other', 'Other', 'categories outside the six families', [])]:
        if key in fam_map:
            page['families'].append({'key': key, 'title': ftitle, 'line': line, 'cards': fam_map[key]})
    if assets.lossy:                # encode every preview now, a few ffmpeg processes at a time
        assets.prefetch([s['path'] for f in page['families'] for c in f['cards'] for v in c['versions']
                         for s in v['sounds']] + [r['path'] for f in page['families'] for c in f['cards']
                                                  for r in c['refs'] if r] +
                        [c['audition'] for f in page['families'] for c in f['cards'] if c.get('audition')])
    page['embed'] = {'codec': codec, 'kbps': assets.kbps, 'lossy': assets.lossy} if embed else None
    return page, assets, warnings


# ------------------------------------------------------------------------------------------------------------ HTML
CSS = """
:root{--ink:#14181f;--ink2:#39424e;--muted:#6a7380;--faint:#959ca6;--line:#e4e6e9;--line2:#eef0f2;--bg:#f5f5f3;
--card:#fff;--tile:#fafaf9;--accent:#27466b;--pass:#1d7048;--passbg:#e7f2ec;--flag:#9b3b2a;--flagbg:#f8ebe7;
--sans:Inter,"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
--mono:"JetBrains Mono","Cascadia Mono","SF Mono",ui-monospace,Consolas,monospace}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1440px;margin:0 auto;padding:0 32px}
header.top{padding-top:44px;padding-bottom:20px}
.topgrid{display:flex;flex-wrap:wrap;gap:24px 40px;align-items:flex-start;margin-top:22px}
.topgrid .ov{flex:0 1 auto;min-width:0;max-width:100%;overflow-x:auto}
.topgrid aside{flex:1 1 280px;min-width:260px}
h1{font-size:30px;line-height:1.15;font-weight:650;letter-spacing:-.015em;margin:0 0 8px}
.lede{color:var(--ink2);font-size:15px;max-width:860px;margin:0}
.meta{color:var(--muted);font-size:12.5px;margin-top:12px;display:flex;flex-wrap:wrap;gap:6px 22px}
.meta b{color:var(--ink2);font-weight:600}
nav.fam{position:sticky;top:0;z-index:5;background:rgba(245,245,243,.94);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);margin-bottom:8px}
nav.fam .wrap{display:flex;gap:4px;overflow-x:auto;padding-top:10px;padding-bottom:10px}
nav.fam a{color:var(--ink2);font-size:13px;font-weight:550;padding:5px 12px;border-radius:999px;white-space:nowrap}
nav.fam a:hover{background:#e9eaec;text-decoration:none}
nav.fam .n{color:var(--faint);font-weight:500;margin-left:6px;font-variant-numeric:tabular-nums}
nav.fam label.lm{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--ink2);
white-space:nowrap;padding-left:16px;cursor:pointer}
nav.fam label.lm input{accent-color:var(--accent);margin:0}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px 28px;margin:0;font-size:12.5px;
color:var(--ink2)}
.legh{margin:0 0 10px;font-size:11px;font-weight:650;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.legend dt{font-weight:600;color:var(--ink)}.legend dd{margin:0 0 2px;color:var(--muted)}
.overview{border-collapse:separate;border-spacing:0;margin:0;font-size:12.5px;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
.overview th,.overview td{padding:5px 14px;text-align:left;border-bottom:1px solid var(--line2);white-space:nowrap}
.overview th{font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);
background:#fbfbfa}
.overview td.num,.overview th.num{text-align:right;font-variant-numeric:tabular-nums}
.overview tr.famrow td{background:#f7f7f6;font-weight:600;color:var(--ink2);font-size:12px}
.overview tr:last-child td{border-bottom:0}
.overview .vsep{border-left:1px solid var(--line)}
section.family{margin:44px 0 0}
section.family>h2{font-size:21px;font-weight:650;letter-spacing:-.01em;margin:0}
section.family>p{color:var(--muted);margin:2px 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px 24px 18px;margin:0 0 18px;
scroll-margin-top:64px}
.chead{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 14px}
.chead h3{font-size:18px;font-weight:650;margin:0;letter-spacing:-.005em}
.slug{font:12px var(--mono);color:var(--faint)}
.badge{display:inline-block;font-size:10.5px;font-weight:650;letter-spacing:.06em;text-transform:uppercase;
padding:2px 7px;border-radius:5px;background:#eceef1;color:var(--ink2);vertical-align:2px}
.badge.pro{background:#e6ecf4;color:var(--accent)}
.desc{color:var(--ink2);margin:4px 0 0;max-width:900px}
.target{color:var(--muted);font-size:12.5px;margin-top:2px}
.target q{color:var(--ink2)}
.cmeta{display:flex;flex-wrap:wrap;gap:6px 20px;margin-top:10px;font-size:12.5px;color:var(--muted);align-items:center}
.cmeta b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.cmeta a{font-weight:550}
.bench{border-collapse:collapse;font-size:12px;margin:14px 0 2px;font-variant-numeric:tabular-nums}
.bench th,.bench td{padding:3px 14px 3px 0;text-align:right;white-space:nowrap}
.bench th{font-weight:600;color:var(--muted);font-size:11px;letter-spacing:.03em;text-transform:uppercase}
.bench td.l,.bench th.l{text-align:left;color:var(--ink2)}
.bench td.h{color:var(--faint);text-align:left;padding-left:6px;white-space:normal}
.bench td.made{font-weight:650;color:var(--ink)}
.benchwrap{display:flex;flex-wrap:wrap;gap:4px 36px}
.benchwrap .vl{font-size:11px;font-weight:650;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);
margin:14px 0 0}
.rows{margin-top:16px;display:flex;flex-direction:column;gap:10px}
.row{display:grid;grid-template-columns:64px minmax(0,1fr);gap:12px;align-items:start}
.row.nolabel{grid-template-columns:minmax(0,1fr)}
.rlabel{font-size:11px;font-weight:650;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);
padding-top:10px;line-height:1.3}
.rlabel span{display:block;color:var(--faint);font-weight:500;letter-spacing:0;text-transform:none;margin-top:2px}
.tiles{display:grid;grid-template-columns:repeat(var(--n),minmax(176px,1fr));gap:10px;overflow-x:auto;
padding-bottom:2px}
.tile{background:var(--tile);border:1px solid var(--line2);border-radius:9px;padding:9px 10px 10px;min-width:0;
display:flex;flex-direction:column;gap:7px}
.tile.ref{background:#f6f7f9;border-style:dashed;border-color:#dfe3e8}
.tile.empty{background:transparent;border:1px dashed var(--line);min-height:60px}
.spec{display:block;width:100%;height:84px;border-radius:5px;background:#fbfbfa;border:1px solid var(--line2);
image-rendering:auto}
.tname{display:flex;align-items:center;justify-content:space-between;gap:8px;min-width:0}
.tname .n{font-weight:650;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tname .v{color:var(--muted);font-weight:500;margin-left:6px}
.pill{font-size:10.5px;font-weight:650;letter-spacing:.04em;padding:1px 7px;border-radius:999px;white-space:nowrap}
.pill.pass{background:var(--passbg);color:var(--pass)}
.pill.no{background:#eff0f2;color:var(--muted)}
.pill.same{background:#fff;color:var(--muted);border:1px solid var(--line)}
.pills{display:flex;gap:4px}
audio{width:100%;height:30px;display:block}
.kv{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px 8px;font-variant-numeric:tabular-nums}
.kv div{min-width:0}
.kv i{display:block;font-style:normal;font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint);
line-height:1.2}
.kv b{display:block;font-weight:550;font-size:12.5px;color:var(--ink);white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.kv.bk{border-top:1px solid var(--line2);padding-top:6px}
.kv.bk i{color:var(--accent);opacity:.8}
.words{font-size:11.5px;color:var(--muted);line-height:1.35}
.flag{font-size:11.5px;color:var(--flag);background:var(--flagbg);border-radius:5px;padding:2px 6px;line-height:1.35}
details.params{margin-top:14px;border-top:1px solid var(--line2);padding-top:10px}
details.params summary{cursor:pointer;font-size:12.5px;font-weight:600;color:var(--ink2);list-style:none}
details.params summary::-webkit-details-marker{display:none}
details.params summary:before{content:"";display:inline-block;width:0;height:0;border-left:5px solid var(--muted);
border-top:4px solid transparent;border-bottom:4px solid transparent;margin-right:8px;vertical-align:1px;
transition:transform .15s}
details.params[open] summary:before{transform:rotate(90deg)}
.ptab{overflow-x:auto;margin-top:10px}
.ptab table{border-collapse:collapse;font:11.5px/1.4 var(--mono);font-variant-numeric:tabular-nums}
.ptab th,.ptab td{padding:2px 14px 2px 0;text-align:right;white-space:nowrap}
.ptab th{font:600 11px var(--sans);color:var(--muted)}
.ptab td.k,.ptab th.k{text-align:left;color:var(--ink2)}
.ptab tr.vl td{font:650 11px var(--sans);color:var(--accent);text-transform:uppercase;letter-spacing:.06em;
padding-top:10px}
.method{font-size:12px;color:var(--muted);max-width:980px;margin:8px 0 0}
.audition{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--muted);margin-top:10px}
.audition audio{width:320px}
.missing{color:var(--faint);font-size:12.5px;padding:10px 0}
.warn{margin:26px 0 0;font-size:12px;color:var(--muted)}
.warn summary{cursor:pointer}
footer{color:var(--faint);font-size:12px;padding:48px 0 56px}
.benchwrap>div{max-width:100%;overflow-x:auto}
@media (max-width:760px){.wrap{padding:0 16px}.row{grid-template-columns:minmax(0,1fr)}.rlabel{padding-top:0}
h1{font-size:24px}.bench td.h{display:none}.card{padding:16px 14px}.topgrid .ov{width:100%}
.tiles{grid-template-columns:repeat(var(--n),minmax(200px,1fr));scroll-snap-type:x mandatory;
scroll-padding-left:2px}.tile{scroll-snap-align:start}}
"""

JS = """(function(){var d=document.querySelectorAll('audio[data-src-of]');for(var i=0;i<d.length;i++){
var o=document.getElementById(d[i].getAttribute('data-src-of'));if(o)d[i].src=o.getAttribute('src');}})();
document.addEventListener('play',function(e){var a=document.getElementsByTagName('audio');
for(var i=0;i<a.length;i++){if(a[i]!==e.target)a[i].pause();}},true);
(function(){var cb=document.getElementById('lm');if(!cb)return;var T=parseFloat(cb.getAttribute('data-target'));
function set(){var a=document.querySelectorAll('audio[data-lufs]');for(var i=0;i<a.length;i++){
var l=parseFloat(a[i].getAttribute('data-lufs'));
a[i].volume=cb.checked&&isFinite(l)?Math.min(1,Math.pow(10,(T-l)/20)):1;}}
cb.addEventListener('change',set);set();})();"""


def _e(s):
    return html.escape(str(s), quote=True)


def _f(v, fmt='{:.2f}', dash='-'):
    return fmt.format(v) if _num(v) is not None else dash


def _pval(v):
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        a = abs(v)
        return f'{v:.0f}' if a >= 1000 else f'{v:.4g}'
    if isinstance(v, (list, tuple)):
        return ', '.join(_pval(q) for q in v[:8]) + (' ...' if len(v) > 8 else '')
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k}: {_pval(q)}' for k, q in list(v.items())[:6]) + '}'
    return str(v)


def _kv(items, cls='kv'):
    cells = ''.join(f'<div title="{_e(t)}"><i>{_e(k)}</i><b>{_e(v)}</b></div>' for k, v, t in items)
    return f'<div class="{cls}">{cells}</div>'


def _tile(s, assets, ref=False):
    if s is None:
        return '<div class="tile empty"></div>'
    c, b = s.get('critic') or {}, s.get('bench')
    name = s['name']
    short = re.sub(r'^.*?_(\d+)$', r'\1', name) if re.search(r'_\d+$', name) and not ref else name
    variant = s.get('variant') or ''
    pill = ''
    if c.get('passes') is True:
        pill = '<span class="pill pass" title="passes the forge critic">pass</span>'
    elif c.get('passes') is False:
        pill = '<span class="pill no" title="does not pass the forge critic">no pass</span>'
    if s.get('same_as'):
        pill = (f'<span class="pills"><span class="pill same" title="the same file as in {_e(s["same_as"])} (identical '
                f'bytes, or unchanged since its benchmark run)">= {_e(s["same_as"])}</span>{pill}</span>')
    img = (f'<img class="spec" loading="lazy" src="{_e(assets.url(s["png"]))}" alt="spectrogram of {_e(name)}">'
           if s.get('png') else '<div class="spec"></div>')
    title = name + (f' ({variant})' if variant else '')
    head = (f'<div class="tname"><span class="n" title="{_e(title)}">{_e(short if variant else name)}'
            f'{f"<span class=v>{_e(variant)}</span>" if variant else ""}</span>{pill}</div>')
    if ref:
        head = f'<div class="tname"><span class="n" title="{_e(s["file"])}">{_e(s["file"])}</span></div>'
    lu = f' data-lufs="{s["lufs"]:.1f}"' if _num(s.get('lufs')) is not None else ''
    audio = f'<audio controls preload="none"{lu}{assets.audio_src(s["path"])}></audio>'
    base = [('length', f"{s['dur_s']:.2f} s" if s['dur_s'] < 10 else f"{s['dur_s']:.1f} s", 'duration of the file'),
            ('peak', _f(s.get('peak_dbfs'), '{:.1f}'), 'sample peak, dBFS'),
            ('LUFS', _f(s.get('lufs'), '{:.1f}'), 'momentary-max loudness (K-weighted 400 ms)')]
    parts = [img, head, audio, _kv(base)]
    if not ref and c:
        t40 = c.get('t40_ms')
        t40s = (f'{t40 / 1000:.2f} s' if t40 >= 1000 else f'{t40:.0f} ms') if _num(t40) is not None else '-'
        crit = [('critic', _f(c.get('total')), 'forge critic total (higher is better)'),
                ('tone', _f(c.get('tone_db'), '{:.1f} dB'), "1/3-octave distance to the forge refs' median (critic)"),
                ('tail t40', t40s, 'time to fall 40 dB below the peak')]
        parts.append(_kv(crit))
    if not ref and b:
        ins = f"{b['inside']} / {b['n_checks']}" if b.get('inside') is not None and b.get('n_checks') else '-'
        bk = [('bench tone', _f(b.get('tone_db'), '{:.1f} dB'), 'held-out benchmark: 1/3-octave distance to the bench '
               'median'), ('bench tail', _f(b.get('t40_ratio'), '{:.2f} x'), 'tail t40 / bench median (1 = same)'),
              ('inside', ins, 'benchmark checks inside the range of real recordings')]
        if b.get('tone_db') is None and b.get('t40_ratio') is None:
            bk = [('phrase z', _f(b.get('phrase_z')), 'CLAP match to independent phrasings (z)'),
                  ('AudioSet', b.get('audioset') or '-', 'AudioSet top label (held-out model)'),
                  ('stereo r', _f(b.get('stereo_corr')), 'L/R correlation')]
        parts.append(_kv(bk, 'kv bk'))
        if b.get('artifacts'):
            parts.append(f'<div class="flag">artifact: {_e(", ".join(b["artifacts"]))}</div>')
    if not ref and c:
        w = c.get('words') or ''
        if w and not w.startswith('clean'):
            parts.append(f'<div class="words" title="the critic\'s words">{_e(w[:220])}</div>')
        elif c.get('sounds_like'):
            parts.append(f'<div class="words" title="nearest descriptions (CLAP)">sounds like: '
                         f'{_e(", ".join(c["sounds_like"][:2]))}</div>')
    return f'<div class="tile{" ref" if ref else ""}">' + ''.join(parts) + '</div>'


def _params_table(card):
    vs = [v for v in card['versions'] if v.get('sounds')]
    if not any(s.get('params') for v in vs for s in v['sounds']):
        return ''
    body = []
    multi = len(vs) > 1
    for v in vs:
        ss = v['sounds']
        keys = []
        for s in ss:
            keys += [k for k in (s.get('params') or {}) if k not in keys]
        head = ''.join(f'<th>{_e(s["name"].rsplit("_", 1)[-1])}{(" " + _e(s["variant"])) if s.get("variant") else ""}'
                       f'</th>' for s in ss)
        rows = [f'<tr><th class="k">parameter</th>{head}</tr>']
        seeds = ''.join(f'<td>{_e(s["seed"] if s.get("seed") is not None else "-")}</td>' for s in ss)
        rows.append(f'<tr><td class="k">seed</td>{seeds}</tr>')
        for k in keys:
            rows.append(f'<tr><td class="k">{_e(k)}</td>' + ''.join(
                f'<td>{_e(_pval(s["params"][k])) if k in (s.get("params") or {}) else "-"}</td>' for s in ss) + '</tr>')
        meta = v.get('meta') or {}
        lab = f'<tr class="vl"><td colspan="{len(ss) + 1}">{_e(v["label"])}</td></tr>' if multi else ''
        method = (f'<p class="method"><b>{_e(v["label"]) + ": " if multi else ""}recipe {_e(meta.get("recipe") or "?")}'
                  f'</b>{". " + _e(meta.get("method")) if meta.get("method") else ""}</p>')
        body.append((method, lab + ''.join(rows)))
    methods = ''.join(m for m, _ in body)
    table = '<table>' + ''.join(t for _, t in body) + '</table>'
    n = sum(len(v['sounds']) for v in vs)
    return (f'<details class="params"><summary>Recipe and parameters <span style="color:var(--faint);font-weight:500">'
            f'({n} sounds)</span></summary>{methods}<div class="ptab">{table}</div></details>')


def _bench_block(card, multi):
    blocks = []
    for v in card['versions']:
        b = v.get('bench')
        if v.get('bench_stale'):
            blocks.append(f'<div><p class="method">{_e(v["label"] + ": " if multi else "")}benchmark numbers withheld: '
                          f'these files changed after the benchmark ran (see the notes at the end)</p></div>')
            continue
        if not b or not b['rows']:
            continue
        rows = ''.join(f'<tr><td class="l">{_e(r[0])}</td><td class="made">{_e(r[1])}</td><td>{_e(r[2])}</td>'
                       f'<td>{_e(r[3])}</td><td class="h">{_e(r[4])}</td></tr>' for r in b['rows'])
        nb = f" ({b['n_bench']} bench refs)" if b.get('n_bench') else ''
        head = (f'<tr><th class="l">held-out bench{_e(nb)}</th><th>made</th><th>forge refs</th><th>bench LOO</th>'
                f'<th></th></tr>')
        src = v.get('bench_from')
        vl = (f'<div class="vl">{_e(v["label"])}'
              + (f' <span style="color:var(--faint);font-weight:500;text-transform:none;letter-spacing:0">(files '
                 f'unchanged since the {_e(src)} benchmark run)</span>' if src and src != v['label'] else '')
              + '</div>') if multi else ''
        blocks.append(f'<div>{vl}<table class="bench">{head}{rows}</table></div>')
    return f'<div class="benchwrap">{"".join(blocks)}</div>' if blocks else ''


def _card(card, page, assets):
    cat = card['cat']
    multi = page['multi']
    pro = cat.startswith('pro_')
    badge = '<span class="badge pro" title="modelled on the designed SFX of the reference launch films">pro</span>' \
        if pro else ''
    newest = next((v for v in reversed(card['versions']) if v.get('sounds')), None)
    meta = (newest or {}).get('meta') or {}
    out = [f'<article class="card" id="{_e(cat)}"><div class="chead"><h3>{_e(humanize(cat))}</h3>{badge}'
           f'<span class="slug">{_e(cat)}</span></div>']
    if card['desc']:
        out.append(f'<p class="desc">{_e(card["desc"])}</p>')
    if meta.get('target'):
        out.append(f'<div class="target">critic target <q>{_e(meta["target"])}</q>'
                   f'{" &middot; " + _e(meta["mode"]) + " mode" if meta.get("mode") else ""}</div>')
    cm = []
    for v in card['versions']:
        if v.get('missing'):
            cm.append(f'<span>{_e(v["label"])}: <b>-</b></span>')
            continue
        crit = [s['critic'].get('passes') for s in v['sounds']]
        known = [c for c in crit if c is not None]
        lab = f'{_e(v["label"])} ' if multi else ''
        if known:
            cm.append(f'<span>{lab}critic pass <b>{sum(known)} / {len(v["sounds"])}</b></span>')
        else:
            cm.append(f'<span>{lab}<b>{len(v["sounds"])}</b> sounds</span>')
    if newest:
        for name, q in newest['links'].items():
            cm.append(f'<a href="{_e(assets.link(q))}">{_e(name)}</a>')
    cm.append(f'<span title="spectrogram time span (all rows), log frequency 30 Hz-20 kHz, 80 dB">span '
              f'<b>{card["span"]:.2f} s</b></span>')
    out.append(f'<div class="cmeta">{"".join(cm)}</div>')
    out.append(_bench_block(card, multi))
    n = max([len(v.get('order', v['sounds'])) for v in card['versions']] + [len(card['refs'])] + [1])
    labelled = multi or bool(card['refs'])
    rows = []
    for v in card['versions']:
        if v.get('missing'):
            inner = f'<div class="missing">no {_e(v["label"])} sounds for this category</div>'
        else:
            tiles = ''.join(_tile(s, assets) for s in v.get('order', v['sounds']))
            inner = f'<div class="tiles" style="--n:{n}">{tiles}</div>'
        if labelled:
            same = v['sounds'] and all(q.get('same_as') for q in v['sounds'])
            sub = (f'<span>{"unchanged" if same else str(len(v["sounds"])) + " sounds"}</span>'
                   if not v.get('missing') else '')
            rows.append(f'<div class="row"><div class="rlabel">{_e(v["label"] if multi else "made")}{sub}</div>'
                        f'{inner}</div>')
        else:
            rows.append(f'<div class="row nolabel">{inner}</div>')
    if card['refs']:
        inner = f'<div class="tiles" style="--n:{n}">' + ''.join(_tile(r, assets, ref=True) for r in card['refs']) \
            + '</div>'
        rows.append(f'<div class="row"><div class="rlabel">ref<span>nearest forge ref</span></div>{inner}</div>')
    out.append('<div class="rows">' + ''.join(rows) + '</div>')
    if card.get('audition'):
        out.append(f'<div class="audition"><span>audition reel (made, nearest ref, ...)</span>'
                   f'<audio controls preload="none"{assets.audio_src(card["audition"])}></audio></div>')
    out.append(_params_table(card))
    out.append('</article>')
    return ''.join(out)


def _overview(page):
    labels = page['labels']
    multi = page['multi']
    has_bench = {lab: any(v.get('bench') for f in page['families'] for c in f['cards'] for v in c['versions']
                          if v['label'] == lab) for lab in labels}
    th = '<th>category</th>'
    for lab in labels:
        pre = f'{_e(lab)} ' if multi else ''
        th += f'<th class="num vsep">{pre}sounds</th><th class="num">{pre}critic pass</th>'
        if has_bench[lab]:
            th += (f'<th class="num" title="CLAP cosine between variants (lower = more varied); the forge refs in '
                   f'brackets">{pre}variety (refs)</th><th class="num" title="tail t40 / bench median">{pre}tail</th>'
                   f'<th class="num" title="1/3-octave distance to the bench median">{pre}tone dB</th>'
                   f'<th class="num" title="median checks inside the range of real recordings">{pre}inside</th>')
    rows = [f'<tr>{th}</tr>']
    ncol = 1 + sum(2 + (4 if has_bench[lab] else 0) for lab in labels)
    for fam in page['families']:
        rows.append(f'<tr class="famrow"><td colspan="{ncol}"><a href="#fam-{_e(fam["key"])}">{_e(fam["title"])}</a>'
                    f'</td></tr>')
        for card in fam['cards']:
            tds = f'<td><a href="#{_e(card["cat"])}">{_e(card["cat"])}</a></td>'
            for lab, v in zip(labels, card['versions']):
                ss = v.get('sounds') or []
                known = [s['critic'].get('passes') for s in ss if s['critic'].get('passes') is not None]
                same = ss and all(s.get('same_as') for s in ss)
                mark = (f' <span style="color:var(--faint)" title="the same files as {_e(ss[0]["same_as"])}">= '
                        f'{_e(ss[0]["same_as"])}</span>') if same else ''
                tds += (f'<td class="num vsep">{mark}{len(ss) if ss else "-"}</td>'
                        f'<td class="num">{f"{sum(known)} / {len(ss)}" if known else "-"}</td>')
                if has_bench[lab]:
                    rmap = {r[0]: r for r in ((v.get('bench') or {}).get('rows') or [])}

                    def cell(key, i=1):
                        r = rmap.get(key)
                        return _e(r[i]) if r else '-'
                    var = cell('variety')
                    if 'variety' in rmap and rmap['variety'][2] != '-':
                        var += f' <span style="color:var(--faint)">({_e(rmap["variety"][2])})</span>'
                    tds += (f'<td class="num">{var}</td><td class="num">{cell("tail t40")}</td>'
                            f'<td class="num">{cell("tone distance")}</td><td class="num">{cell("inside")}</td>')
            rows.append(f'<tr>{tds}</tr>')
    return f'<table class="overview">{"".join(rows)}</table>'


def render_html(page, assets, warnings):
    labels = page['labels']
    multi = page['multi']
    title = page['title'] or ('earshot forge: before and after' if multi else 'earshot forge: sound gallery')
    n_cards = sum(len(f['cards']) for f in page['families'])
    passes = ', '.join(f'{lab} {a} / {b}' if multi else f'{a} / {b}' for lab, (a, b) in page['n_pass'].items())
    meta = [f'<span><b>{n_cards}</b> categories</span>', f'<span><b>{page["n_sounds"]}</b> sounds</span>',
            f'<span>critic pass <b>{_e(passes)}</b></span>']
    for lab, b in page['bench'].items():
        meta.append(f'<span>benchmark{" " + _e(lab) if multi else ""} <b>{_e(b.get("tag") or "")}</b> '
                    f'({_e(b.get("created") or "")}, {_e(os.path.basename(b["file"]))})</span>')
    if page['n_refs']:
        meta.append(f'<span><b>{page["n_refs"]}</b> forge references (licensed: keep this page private)</span>')
    emb = page.get('embed') or {}
    if emb.get('lossy'):
        meta.append(f'<span title="the players play compressed previews; every number and spectrogram on the page is '
                    f'measured on the original WAV files">audio <b>{_e(emb["codec"].upper())} {emb["kbps"]} kbps</b> '
                    f'previews (numbers from the WAVs)</span>')
    meta.append(f'<span>built {_e(time.strftime("%Y-%m-%d %H:%M"))}</span>')
    lede = ('Every sound on this page was computed by a recipe of code (no recorded audio inside) and tuned by the '
            'forge, which listens to its own renders with earshot and compares them to reference recordings. ')
    lede += ('Each category shows the versions as rows, variant under variant, so before and after read top to bottom.'
             if multi else 'Each category shows its variants side by side.')
    nav = ''.join(f'<a href="#fam-{_e(f["key"])}">{_e(f["title"])}<span class="n">{len(f["cards"])}</span></a>'
                  for f in page['families'])
    lufs = [s['lufs'] for f in page['families'] for c in f['cards'] for v in c['versions'] for s in v['sounds']
            if _num(s.get('lufs')) is not None]
    if lufs:
        target = float(np.clip(np.percentile(lufs, 10), -30, -14))
        nav += (f'<label class="lm" title="players turn louder sounds down to {target:.0f} LUFS (momentary max); '
                f'quieter ones play as they are"><input type="checkbox" id="lm" data-target="{target:.1f}" checked>'
                f'match loudness</label>')
    legend = [('critic', 'the forge critic\'s total and pass verdict for that sound (its own refs, never the bench); '
               'totals compare variants within a category, not across categories (each has its own refs and mode)'),
              ('tone / tail t40', 'critic-side: 1/3-octave distance to the forge refs\' median; time to fall 40 dB'),
              ('LUFS', 'momentary-max loudness, K-weighted 400 ms windows (the forge\'s loudness number)')]
    if page['bench']:
        legend += [('bench', 'held-out benchmark (eval/forge/bench.py): reference recordings the forge never heard; '
                    'made vs forge refs vs bench leave-one-out'),
                   ('inside', 'number of benchmark checks within the range real recordings reach'),
                   ('-', 'no benchmark number: the category has no bench split for it, or the files changed after the '
                    'benchmark ran (the notes at the end list them)')]
    if multi:
        legend += [(f'= {labels[0]}', 'a row or sound marked like this holds the same files as that version '
                    '(identical bytes, or unchanged at the path its benchmark measured): not redone yet')]
    legend += [('spectrogram', f'log frequency 30 Hz to 20 kHz (bottom to top; faint lines at 100 Hz, 1 kHz, 10 kHz), '
                f'{FLOOR_DB:.0f} dB range, from the onset; one time span per card so lengths compare')]
    leg = ('<h4 class="legh">How to read it</h4><dl class="legend">'
           + ''.join(f'<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>' for k, v in legend) + '</dl>')
    body = [f'<header class="top wrap"><h1>{_e(title)}</h1><p class="lede">{_e(lede)}</p>'
            f'<div class="meta">{"".join(meta)}</div><div class="topgrid"><div class="ov">{_overview(page)}</div>'
            f'<aside>{leg}</aside></div></header>',
            f'<nav class="fam"><div class="wrap">{nav}</div></nav><main class="wrap">']
    for fam in page['families']:
        body.append(f'<section class="family" id="fam-{_e(fam["key"])}"><h2>{_e(fam["title"])}</h2>'
                    f'<p>{_e(fam["line"])}</p>')
        body += [_card(c, page, assets) for c in fam['cards']]
        body.append('</section>')
    if warnings:
        body.append('<details class="warn"><summary>' + _e(f'{len(warnings)} note(s) from the build') + '</summary><ul>'
                    + ''.join(f'<li>{_e(w)}</li>' for w in warnings) + '</ul></details>')
    body.append('<footer>Made with earshot gallery. Audio: computed by forge recipes'
                + (', plus licensed forge-side reference recordings for A/B' if page['n_refs'] else '')
                + (f' (played as {emb["codec"].upper()} {emb["kbps"]} kbps previews of the WAV files)'
                   if emb.get('lossy') else '')
                + '. Bench references are never shown here.</footer></main>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(title)}</title><style>{CSS}</style></head><body>{"".join(body)}'
            f'<script>{JS}</script></body></html>')


def gallery(folder=None, out='gallery.html', results=None, refs=None, compare=None, embed=False, codec='flac',
            sub='final', title=None, max_refs=6, compressed=False, kbps=None, max_mb=25.0, alias=None):
    """build the page; `folder` (a forge output tree) or `compare` (['r2:GLOB', 'r3:GLOB']); returns a summary dict.
    compressed=True: one self-contained page with lossy previews (codec 'mp3' unless codec is 'opus' / 'vorbis')"""
    t0 = time.time()
    if isinstance(results, str):
        results = [results]
    codec = codec or 'flac'
    if codec not in CODECS:
        raise ValueError(f'codec must be one of {", ".join(CODECS)}, not {codec!r}')
    if compressed:
        embed = True
        codec = codec if codec in LOSSY else 'mp3'
    if kbps is not None and not (codec in LOSSY and embed):
        raise ValueError('kbps applies to a lossy embedded codec only (--embed-compressed, or --embed-codec mp3 / opus '
                         '/ vorbis with --embed)')
    if kbps is not None and not 16 <= int(kbps) <= 320:
        raise ValueError(f'kbps must be between 16 and 320, not {kbps}')
    if compare:
        versions = compare_versions(compare, alias)
    elif folder:
        al = parse_alias(alias)
        versions = [(sub, {al.get(c, c): d for c, d in discover(folder, sub).items()})]
    else:
        raise ValueError('give a folder of forge results or --compare LABEL:GLOB ...')
    if refs == 'bench':
        raise ValueError("bench references are held out: they never go on a page (use --refs forge)")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    page, assets, warnings = build(versions, out, results, refs, embed, codec, title, max_refs, kbps)
    if embed and page['n_refs']:
        warnings.insert(0, f"this single file carries {page['n_refs']} licensed forge reference recording(s) inside "
                           f"it: keep it private (build it without --refs to share it)")
    doc = render_html(page, assets, warnings)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    pruned = assets.prune()
    size = os.path.getsize(out) / 1e6
    if embed and max_mb and size > max_mb:
        warnings.append(f'the page is {size:.1f} MB, over {max_mb:g} MB: lower --kbps, use --embed-codec opus, or '
                        f'build fewer categories')
    res = {'html': out.replace(chr(92), '/'), 'assets': assets.dir.replace(chr(92), '/'),
           'versions': {lab: len(m) for lab, m in versions},
           'categories': sum(len(f['cards']) for f in page['families']), 'sounds': page['n_sounds'],
           'refs': page['n_refs'], 'images': {'made': assets.made, 'reused': assets.reused, 'pruned': pruned},
           'embedded': bool(embed), 'codec': (f'{codec} {assets.kbps} kbps' if assets.lossy else codec) if embed
           else None, 'size_mb': round(size, 2), 'critic_pass': page['n_pass'],
           'seconds': round(time.time() - t0, 1), 'warnings': warnings}
    from pathlib import Path
    res['url'] = Path(out).as_uri()
    nc, ns = res['categories'], res['sounds']
    lines = [f"gallery: {nc} {'category' if nc == 1 else 'categories'}, {ns} sound{'' if ns == 1 else 's'}"
             + (f", {res['refs']} refs" if res['refs'] else '')
             + f" -> {res['html']} ({res['size_mb']} MB{', embedded ' + res['codec'] if embed else ''}, "
             f"{res['seconds']} s)",
             f"  versions: {', '.join(f'{k} ({v})' for k, v in res['versions'].items())}",
             f"  thumbnails: {assets.made} new, {assets.reused} cached, {pruned} pruned in {res['assets']}"]
    if embed:
        res['embedded_mb'] = {'audio': round(assets.audio_bytes * 4 / 3 / 1e6, 2),
                              'images': round(assets.image_bytes * 4 / 3 / 1e6, 2)}
        lines.append(f"  embedded (base64): audio {res['embedded_mb']['audio']} MB, images "
                     f"{res['embedded_mb']['images']} MB"
                     + (f"; {assets.deduped} identical file(s) embedded once" if assets.deduped else '')
                     + (f"; previews {assets.encoded} encoded, the rest from the cache" if assets.lossy else ''))
    lines.append(f"  open: {res['url']}")
    lines += [f'  note: {w}' for w in warnings[:12]] + ([f'  ... {len(warnings) - 12} more notes'] if len(warnings) > 12
                                                          else [])
    res['text'] = '\n'.join(lines)
    return res


def add_cli(sub):
    from .cli import common
    from .report import emit
    p = sub.add_parser('gallery', help='one static HTML page of forge results: players, spectrograms, numbers, recipes')
    p.add_argument('folder', nargs='?',
                   help='forge output tree (<category>/final/*.wav), one category or one final dir')
    p.add_argument('-o', dest='html', default='gallery.html',
                   help='output HTML (default gallery.html); thumbnails go to <stem>_files/ next to it')
    p.add_argument('--results', nargs='+', metavar='[LABEL:]JSON',
                   help='benchmark results from eval/forge/bench.py (with --compare: LABEL:JSON per version)')
    p.add_argument('--refs', metavar='forge|DIR',
                   help="A/B row of forge-side references: 'forge' = the registry's forge split, or "
                   "DIR/<category>/[forge/]*.wav (bench refs are refused)")
    p.add_argument('--compare', nargs='+', metavar='LABEL:GLOB', help='versions as rows, e.g. '
                   'r2:out/forge_v2/*/final_r2 r3:out/forge_v2/*/final (quote the globs)')
    p.add_argument('--sub', default='final', help='subfolder per category (default final; e.g. final_r2)')
    p.add_argument('--alias', nargs='+', metavar='OLD=NEW', help='show folder category OLD as NEW (e.g. v1 recipe '
                   'folders: keypress=typing_key whoosh=whoosh_fast)')
    p.add_argument('--embed', action='store_true', help='embed audio and images as base64 (one portable file)')
    p.add_argument('--embed-compressed', dest='compressed', action='store_true',
                   help='one small self-contained page: lossy previews via ffmpeg (MP3 192 kbps by default), JPEG '
                   'thumbnails, no links to local files')
    p.add_argument('--embed-codec', dest='codec', choices=list(CODECS), default=None,
                   help='codec of embedded audio: flac (default with --embed: lossless, about half the size), wav, or '
                   'with --embed-compressed mp3 (default: plays everywhere), opus or vorbis (Ogg)')
    p.add_argument('--kbps', type=int, help='bit rate of lossy previews (default mp3 192, opus 128, vorbis 160)')
    p.add_argument('--max-mb', type=float, default=25.0, help='note when an embedded page is larger (default 25)')
    p.add_argument('--title', help='page title')
    p.add_argument('--max-refs', type=int, default=6, help='at most this many A/B references per category')
    common(p)

    def run(a):
        if not a.folder and not a.compare:
            raise ValueError('give a folder of forge results or --compare LABEL:GLOB ...')
        return emit(gallery(a.folder, a.html, a.results, a.refs, a.compare, a.embed, a.codec, a.sub, a.title,
                            a.max_refs, a.compressed, a.kbps, a.max_mb, a.alias), a) and 0
    p.set_defaults(func=run)


from ._callable import callable_module  # noqa: E402
callable_module(__name__)
