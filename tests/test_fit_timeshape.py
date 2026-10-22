"""Time shape of fits (earshot 0.4, fit/auto.py + fit/modal.py), on synthetic sounds only:
- the post stage's gate: a gated pulse keeps its hard stop and its gaps, a cut never reaches into the next onset,
  knobs that move time move the gate, a final stop stays shut in a take;
- character()'s glide gate fires on a 2-3 semitone chirp of 40 ms, not on a steady tone;
- a bed under a sound (a film's music in the pre-roll) is found, stripped before the fit and kept apart;
- modal / hybrid fits jitter their takes half as much per unit of vary (TAKE_VARY)."""
import numpy as np

from earshot.forge.fit import auto
from earshot.forge.fit import quality as Q

SR = 48000


def _db(v):
    return 10 * np.log10(np.mean(np.asarray(v, float) ** 2) + 1e-20)


def _pulse(n, t0, f=2000.0, tau=0.03, stop=None, amp=0.5):
    """a damped tone from t0 (s); stop: the time (s) it is cut to silence (a gate), None = it decays freely"""
    t = np.arange(n) / SR - t0
    y = (t >= 0) * amp * np.exp(-np.clip(t, 0, None) / tau) * np.sin(2 * np.pi * f * np.clip(t, 0, None))
    if stop is not None:
        y[int(stop * SR):] = 0.0
    return y


def _gated_run(n_s=0.16, gate=True):
    """three pulses 50 ms apart, each cut 12 ms after its onset (gate=True) or ringing on (gate=False)"""
    n = int(n_s * SR)
    return sum(_pulse(n, t0, f, stop=(t0 + 0.012) if gate else None)
               for t0, f in ((0.01, 2000.0), (0.06, 2300.0), (0.11, 2600.0)))


# ------------------------------------------------------------------------------------------------------ gate
def test_gate_restores_a_hard_stop_and_spares_the_next_onset():
    x, ring = _gated_run(), _gated_run(gate=False)
    plain = auto.fit_post(x, [ring], gate=False)
    post = auto.fit_post(x, [ring])
    assert 'gate' in post and 'gate' not in plain
    y0, y = auto.apply_post(ring, plain), auto.apply_post(ring, post)
    gap = slice(int(0.03 * SR), int(0.058 * SR))                    # between the first and second pulse
    assert _db(y[gap]) < _db(y0[gap]) - 20                          # the ring between the pulses is cut
    on = slice(int(0.0605 * SR), int(0.064 * SR))                   # the second pulse's first 3.5 ms
    assert abs(_db(y[on]) - _db(y0[on])) < 1.5                      # ... and its onset is not
    fall = lambda v: max(10 * np.log10(np.mean(v[i:i + 48] ** 2) + 1e-20)
                         - 10 * np.log10(np.mean(v[i + 240:i + 288] ** 2) + 1e-20) for i in range(480, 1200, 24))
    assert fall(y) > 30 > fall(y0)                                  # a 5 ms fall like the original's
    assert Q.quality(x, y)['excess_db'] < Q.quality(x, y0)['excess_db']


def test_gate_is_cut_only_and_holds_a_final_stop():
    n = int(0.12 * SR)
    x, ring = _pulse(n, 0.01, stop=0.04), _pulse(n, 0.01)
    post = auto.fit_post(x, [ring])
    pts = np.asarray(post['gate']['pts'])
    assert pts[:, 1].max() <= 0.0 and pts[-1, 1] < -30             # cuts only; the stop is still shut at the end
    g = auto._gate_gain(n + SR // 10, post['gate'])                 # a take that rings past the recording's end
    assert 20 * np.log10(g[-1]) < -30


def test_no_gate_on_a_freely_decaying_strike_or_a_steady_noise():
    n = int(0.15 * SR)
    x = _pulse(n, 0.01)
    assert 'gate' not in auto.fit_post(x, [x.copy()])
    r = np.random.default_rng(3)
    a, b = (0.1 * r.standard_normal(n) for _ in range(2))           # two takes of one noise: flicker, not gaps
    assert 'gate' not in auto.fit_post(a, [b])


def test_gate_moves_with_the_knobs_that_move_time():
    m = {'kind': 'modal', 'strikes': [{'t': 0.01}, {'t': 0.05}]}
    f = auto._gate_map(m, {'spacing': 2.0})
    t, g = f(np.array([0.02, 0.03, 0.06, 0.07]), np.array([-20.0, 0.0, -20.0, 0.0]))
    assert np.allclose(t[:2], [0.02, 0.03]) and np.allclose(t[-2:], [0.10, 0.11])   # cuts after the second strike
    assert np.allclose(g[-2:], [-20.0, 0.0]) and np.all(np.diff(t) > 0)            # move with it
    assert auto._gate_map(m, {'spacing': 1.0, 'damping': 0.5}) is None   # a longer decay does not move a gate
    s = auto._gate_map({'kind': 'sines', 'onset': 0.01}, {'stretch': 1.5})
    assert np.allclose(s(np.array([0.01, 0.03]), np.zeros(2))[0], [0.01, 0.04])


def _gain_db(gate, tmap, t_s, n=int(0.3 * SR)):
    a = auto._gate_gain(n, gate, tmap)
    return np.array([20 * np.log10(a[int(round(v * SR))] + 1e-12) for v in np.atleast_1d(t_s)])


def test_spacing_moves_the_reopening_with_the_next_strike():
    """review: a gap cut that reopens just before the second strike, at spacing 0.8, dropped the cut's end and its
    reopening and bridged -40 dB to the next corner straight across the moved strike (about -25 dB on it); at 1.25 it
    reopened at the old place and the stretched gap rang on. The reopening now moves with the strike it opens for,
    the cut before it holds"""
    m = {'kind': 'modal', 'strikes': [{'t': 0.010}, {'t': 0.100}]}
    gate = {'pts': [[0.012, 0.0], [0.03, -40.0], [0.0985, -40.0], [0.099, 0.0], [0.15, 0.0], [0.16, -60.0]]}
    short = auto._gate_map(m, {'spacing': 0.8})                     # the second strike at 0.082
    on, gap, after = _gain_db(gate, short, [0.0825, 0.0835]), _gain_db(gate, short, [0.05, 0.075]), \
        _gain_db(gate, short, [0.15])
    assert on.min() > -1.0 and gap.max() < -35.0, (on, gap)        # the strike whole, the shortened gap still cut
    assert after[0] < -40                                            # the second strike's own cut moved with it
    long = auto._gate_map(m, {'spacing': 1.25})                     # the second strike at 0.1225
    assert _gain_db(gate, long, [0.11, 0.119]).max() < -35.0         # the stretched gap held shut
    assert _gain_db(gate, long, [0.1225, 0.124]).min() > -1.0        # and open on the strike


def test_gated_modal_run_keeps_its_later_strikes_under_spacing():
    x = _gated_run()
    m = auto.fit(x, kind='modal', fast=True)
    assert 'gate' in m['post'] and len(m['strikes']) >= 3
    t = [s['t'] for s in m['strikes']]

    def peak(y, t0):
        return 20 * np.log10(np.abs(y[int(t0 * SR):int((t0 + 0.006) * SR)]).max() + 1e-12)
    y1 = auto.render(m, seed=0)
    for sp in (0.8, 1.25):
        y = auto.render(m, seed=0, spacing=sp)
        tn = t[0] + (t[1] - t[0]) * sp
        assert peak(y, tn) > peak(y1, t[1]) - 3.0, sp               # the moved second strike is not under a cut


def test_morph_moves_each_gate_onto_the_morphs_timeline():
    """review: the gates of two fits mixed on their own times, so a cut of one landed on the morph's strike and a
    hard stop of one cut the morph's tail before its morphed end; each gate is now warped through the morph's
    anchors (_morph_maps) before they mix"""
    a = {'kind': 'modal', 'strikes': [{'t': 0.01}, {'t': 0.10}], 'length_s': 0.25}
    b = {'kind': 'modal', 'strikes': [{'t': 0.01}, {'t': 0.16}], 'length_s': 0.45}
    pa = {'eq_db': [0.0] * 28, 'gain_db': 0.0,
          'gate': {'pts': [[0.012, 0.0], [0.03, -60.0], [0.0985, -60.0], [0.099, 0.0], [0.2, 0.0], [0.201, -80.0]]}}
    pb = {'eq_db': [0.0] * 28, 'gain_db': 0.0,
          'gate': {'pts': [[0.012, 0.0], [0.03, -60.0], [0.1585, -60.0], [0.159, 0.0], [0.4, 0.0], [0.401, -80.0]]}}
    p = auto._mix_post(pa, pb, 0.5, auto._morph_maps(a, b, 0.5))
    assert _gain_db(p['gate'], None, [0.1305, 0.132], n=int(0.5 * SR)).min() > -1.0    # the morphed strike (0.13)
    assert _gain_db(p['gate'], None, [0.06, 0.12], n=int(0.5 * SR)).max() < -50.0      # the morphed gap
    assert _gain_db(p['gate'], None, [0.25], n=int(0.5 * SR))[0] > -1.0                # no stop before ~0.3
    assert _gain_db(p['gate'], None, [0.34], n=int(0.5 * SR))[0] < -60.0               # stopped after it
    raw = auto._mix_post(pa, pb, 0.5)                                                  # (the old mix, on raw times)
    assert _gain_db(raw['gate'], None, [0.131], n=int(0.5 * SR))[0] < -20.0


def test_gesture_length_clamps_the_pre_roll_about_the_tones_onset():
    """review: a gesture with tones stretches about their first strike (its onset stays at t_on), but the attack
    clamp scaled its pre-onset window about t = 0, so at length 1.25 it took 15 ms of the real attack for pre-roll and
    cut it to the stripped bed's -200 dB; scaled about the anchor, the attack stays"""
    t_on, n = 0.06, int(0.15 * SR)
    r = np.random.default_rng(1)
    y = np.zeros(n)
    y[:int(t_on * SR)] = 1e-3 * r.standard_normal(int(t_on * SR))   # leaked pre-roll
    y += _pulse(n, t_on, f=1500.0, tau=0.03)
    pre = {'t_on': t_on, 'db': [-200.0] * int(t_on / auto.PRE_HOP), 'attack_db': [], 'v': 2,
           'hp_hz': auto.PRE_HP_HZ}
    att = slice(int((t_on + 0.001) * SR), int((t_on + 0.015) * SR))
    old = auto._pre_clamp(y, pre, scale=1.25)                       # about 0: the attack taken for pre-roll
    new = auto._pre_clamp(y, pre, scale=1.25, anchor=t_on)
    assert _db(old[att]) < _db(y[att]) - 10
    assert abs(_db(new[att]) - _db(y[att])) < 1.0
    lead = slice(int(0.005 * SR), int(0.05 * SR))
    assert _db(new[lead]) < _db(y[lead]) - 15                       # the pre-roll is still clamped
    assert np.array_equal(auto._pre_clamp(y, pre, scale=1.0, anchor=t_on), auto._pre_clamp(y, pre))


def test_render_passes_the_gesture_anchor_to_the_post_stage(monkeypatch):
    from earshot.forge.fit import gesture
    n = int(0.25 * SR)
    x = _pulse(n, 0.04, f=1800.0, tau=0.05) + 0.02 * np.random.default_rng(2).standard_normal(n) * (
        np.arange(n) >= 0.04 * SR)
    m = gesture.fit(x, fast=True)
    assert m['tones'], m['fit']
    m['post'] = {'eq_db': [0.0] * 28, 'gain_db': 0.0}
    seen = []
    real = auto.apply_post
    monkeypatch.setattr(auto, 'apply_post', lambda y, post, *a: seen.append(a) or real(y, post, *a))
    auto.render(m, seed=0, length=1.25)
    assert seen[-1][0] == 1.25 and seen[-1][-1] == float(m['tones']['onset'])


def test_fitted_modal_run_keeps_its_gaps_and_morphs():
    x = _gated_run()
    m = auto.fit(x, kind='modal', fast=True)
    assert 'gate' in m['post']
    y = auto.render(m, seed=0)
    gap = slice(int(0.03 * SR), int(0.058 * SR))
    assert _db(y[gap]) < _db(y) - 25
    mm = auto.morph(m, m, 0.5)                                      # the gate survives a morph (in dB, mixed)
    assert np.allclose(np.asarray(mm['post']['gate']['pts'])[:, 1].min(),
                       np.asarray(m['post']['gate']['pts'])[:, 1].min(), atol=0.11)


def test_modal_noise_does_not_rise_ahead_of_a_strike():
    from earshot.forge.fit import modal
    r = np.random.default_rng(2)
    n, o = int(0.1 * SR), int(0.03 * SR)
    x = 1e-4 * r.standard_normal(n)
    x[o:] += 0.4 * r.standard_normal(n - o) * np.exp(-np.arange(n - o) / SR / 0.003)       # a noisy contact
    x += _pulse(n, 0.03, f=2500.0, tau=0.01)
    old, new = modal.fit(x, passes=1, pre_cap_s=0.0), modal.fit(x, passes=1)
    pre = slice(o - int(0.0015 * SR), o - int(0.0002 * SR))
    assert _db(modal.render(new, 0)[pre]) < _db(modal.render(old, 0)[pre]) - 3


# ------------------------------------------------------------------------------------------------------ glide
def _chirp(st=2.5, dur=0.04, f0=900.0, n_s=0.08):
    t = np.arange(int(n_s * SR)) / SR
    f = f0 * 2 ** (np.clip(t, 0, dur) / dur * st / 12)
    return 0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(t / 0.002, 1.0) * np.exp(-t / 0.05)


def test_character_sees_a_short_glide_of_two_semitones():
    c = auto.character(_chirp())
    assert c['gliding'] and c['class'] == 'glide' and 0.15 < c['glide_oct'] < 0.3, c
    steady = auto.character(_chirp(st=0.0))
    assert not steady['gliding'] and abs(steady['glide_oct']) < 0.05


# ------------------------------------------------------------------------------------------------------ bed
def _bed(n, seed=1, db=-20.0):
    """a steady 'music' bed: two partials and a little noise, `db` re a 0.5-amplitude strike"""
    t = np.arange(n) / SR
    r = np.random.default_rng(seed)
    y = np.sin(2 * np.pi * 410 * t) + 0.6 * np.sin(2 * np.pi * 655 * t + 1.0) + 0.3 * r.standard_normal(n)
    return y / np.sqrt(np.mean(y ** 2)) * 0.5 / np.sqrt(2) * 10 ** (db / 20)


def test_bed_is_found_stripped_and_kept_apart():
    n = int(0.2 * SR)
    strike = _pulse(n, 0.06, f=3000.0, tau=0.02)
    bed = _bed(n)
    x = strike + bed
    bd = auto.bed_of(x)
    assert bd is not None and abs(bd['t_on'] - 0.06) < 0.002
    assert abs(bd['level_db'] - (_db(bed) - Q.peak_rms_db(x))) < 1.5
    s = auto.strip_bed(x, bd)
    pre = slice(int(0.005 * SR), int(0.055 * SR))
    assert np.max(np.abs(s[pre])) == 0.0              # the pre-roll is silent: the fit starts at the onset
    body = slice(int(0.06 * SR), int(0.08 * SR))
    assert abs(_db(s[body]) - _db(strike[body])) < 1.0              # the strike survives
    tail = slice(int(0.15 * SR), n)
    assert _db(s[tail]) < _db(bed[tail]) - 15                       # the bed under the tail is gone
    assert auto.bed_of(strike + 1e-5 * np.random.default_rng(0).standard_normal(n)) is None   # a clean clip
    assert auto.bed_of(_pulse(n, 0.004) + bed) is None              # no pre-roll to read a bed from


def test_fit_of_a_bedded_clip_fits_the_sound_and_keeps_the_bed_apart():
    n = int(0.2 * SR)
    x = _pulse(n, 0.06, f=3000.0, tau=0.02) + _bed(n)
    m = auto.fit(x, kind='modal', fast=True)
    assert m['post']['bed']['level_db'] > -30 and m['post']['bed']['n'] == n
    pre = slice(int(0.005 * SR), int(0.05 * SR))
    dry, wet = auto.render(m, seed=0, bed=False), auto.render(m, seed=0)
    assert _db(dry[pre]) < _db(x[pre]) - 30                         # the model: the sound alone
    assert abs(_db(wet[pre]) - _db(x[pre])) < 3.0                   # by default with its bed back, like the original
    b = auto.bed_noise(m['post']['bed'], n + SR // 10, seed=1)      # a take that rings past the recording: no bed there
    assert np.max(np.abs(b[n:])) == 0.0 and abs(_db(b[:n]) - _db(_bed(n))) < 2.0
    assert np.max(np.abs(auto.render(m, seed=1, vary=0.2, bed=False)[:int(0.05 * SR)])) < 1e-3       # a dry take


def test_auto_fit_scores_bedded_candidates_against_the_original():
    n = int(0.2 * SR)
    x = _pulse(n, 0.06, f=3000.0, tau=0.02) + _bed(n)
    m = auto.fit(x, candidates=['modal', 'spectral'], fast=True)
    assert m['fit_engine']['bed']['t_on'] > 0.05 and 'bed' in m['post']
    q = Q.quality(x, auto.render(m, seed=0))
    assert q['mel_db'] < 6 and q['excess_db'] < 1.5


def test_cli_keep_bed(tmp_path, wav):
    import json
    from earshot.cli import main
    n = int(0.2 * SR)
    src = wav(_pulse(n, 0.06, f=3000.0, tau=0.02) + _bed(n), 'bedded.wav')
    for flag, want in (([], True), (['--keep-bed'], False)):
        out = tmp_path / ('kept' if flag else 'stripped') / 'b_fit.py'
        assert main(['fit', src, '-o', str(out), '--kind', 'modal', '--fast', '--quiet'] + flag) == 0
        m = json.load(open(out.with_suffix('.json')))
        assert bool((m.get('post') or {}).get('bed')) is want


# ------------------------------------------------------------------------------------------------------ takes
def test_modal_takes_jitter_at_the_scaled_vary():
    x = _pulse(int(0.12 * SR), 0.01) + _pulse(int(0.12 * SR), 0.01, f=3300.0, tau=0.01)
    m = auto.fit(x, kind='modal', fast=True)
    assert m['vary_scale'] == auto.TAKE_VARY['modal']
    plain = {k: v for k, v in m.items() if k != 'vary_scale'}
    assert np.array_equal(auto.render(m, seed=3, vary=0.3), auto.render(plain, seed=3, vary=0.15))
    assert np.array_equal(auto.render(m, seed=0), auto.render(plain, seed=0))   # the fit itself is untouched
    s = auto.fit(x, kind='sines', fast=True)
    assert 'vary_scale' not in s


def test_the_choice_is_confirmed_on_more_seeds():
    r = np.random.default_rng(11)
    n, on = int(0.1 * SR), int(0.01 * SR)
    burst = 0.2 * r.standard_normal(n) * (np.arange(n) >= on) * np.exp(-np.clip(np.arange(n) - on, 0, None) / SR / 0.01)
    x = burst + _pulse(n, 0.01, f=2500.0, tau=0.02, amp=0.3)
    m = auto.fit(x, candidates=['modal', 'spectral'], fast=True, bed=False)
    rows = {row['candidate']: row for row in m['fit_engine']['candidates']}
    assert all('score2' in row for row in rows.values())            # both are among the best CONFIRM
    k = m['fit_engine']['chosen']
    assert k == min(rows, key=lambda c: rows[c]['score2'])          # chosen on the mean, not on seed 0
    more = [Q.score(Q.quality(x, auto.render(m, seed=s))) for s in auto.CONFIRM_SEEDS]
    assert abs(rows[k]['score2'] - np.mean([rows[k]['score']] + more)) < 2e-3
    off = auto.fit(x, candidates=['modal', 'spectral'], fast=True, bed=False, confirm=0)
    assert not any('score2' in row for row in off['fit_engine']['candidates'])


def test_a_post_stage_without_04_fields_still_renders():
    x = _gated_run()
    m = auto.fit(x, kind='modal', fast=True, bed=False)
    old = dict(m, post={k: v for k, v in m['post'].items() if k in ('eq_db', 'gain_db', 'pre')})
    old['post']['pre'] = {k: v for k, v in old['post']['pre'].items() if k != 'v'}
    old.pop('vary_scale', None)
    y = auto.render(old, seed=2, vary=0.2)
    assert np.array_equal(y, auto.render(old, seed=2, vary=0.2)) and np.isfinite(y).all()


# ------------------------------------------------------------------------------------------------------ space
def test_space_layer_is_rendered_in_the_side_only():
    from earshot.forge.fit import space
    n = int(0.3 * SR)
    x = np.stack([_pulse(n, 0.01, f=1500.0, tau=0.05)] * 2, 1)
    m = auto.fit(x, kind='modal', fast=True, bed=False, space=False)
    assert 'space' not in m['post']
    sp = {'kind': 'space', 'version': 1, 'predelay': 0.002, 'build': 0.01, 'bands': list(space.BANDS),
          'rt60': [0.3] * len(space.BANDS), 'side_db': [-6.0] * len(space.BANDS), 'tail_corr': 0.2}
    wet = dict(m, post=dict(m['post'], space=sp))
    y0, y1 = (auto._render(v, seed=1, final=False) for v in (m, wet))   # before the final fade and limit
    assert y1.shape == (n, 2)
    assert np.allclose(y1.mean(1), np.asarray(y0).reshape(n, -1).mean(1), atol=1e-9)   # the mono sum is the fit's
    assert space.tail_corr(y1) < 0.7 < space.tail_corr(y0)                               # the tail is wide
    assert len(auto.render(wet, seed=1, vary=0.2)) > n                                   # a take lets it ring out
    y = auto.render(wet, seed=1)                                        # the render: the same, faded in and limited
    assert np.abs(y).max() <= 10 ** (auto.PEAK_DB / 20) + 1e-12 and y[0].max() == 0.0
    g = min(1.0, 10 ** (auto.PEAK_DB / 20) / np.abs(y1).max())
    assert np.allclose(y[200:], y1[200:] * g, atol=1e-12)


# ------------------------------------------------------------------------------------------------------ plans
def test_long_sounds_try_the_gesture_model_after_sines():
    for cls in auto.LONG_PLANS:
        p = auto.plan({'class': cls, 'dur_s': 1.5})
        assert len(p) == 4 and p == auto.LONG_PLANS[cls][:4]
        if 'gesture' in p:
            assert p.index('gesture') == p.index('sines') + 1     # it takes the sines candidate's partials
    for cls in auto.PLANS:                                        # short sounds: every candidate, gesture after sines
        p = auto.plan({'class': cls, 'dur_s': 0.3})
        assert set(p) == set(auto.ALL) and p.index('gesture') > p.index('sines')
    assert auto.plan({'class': 'gesture', 'dur_s': 0.5})[0] == 'spectral'


# ------------------------------------------------------------------------------------------------------ noise clamp
def test_noise_clamp_puts_no_energy_under_its_high_pass():
    from scipy import signal
    r = np.random.default_rng(5)
    n, on = int(0.06 * SR), int(0.01 * SR)
    hiss = signal.sosfilt(signal.butter(8, 3000, 'highpass', fs=SR, output='sos'), r.standard_normal(n))
    y = 0.3 * hiss * (np.arange(n) >= on) * np.exp(-np.clip(np.arange(n) - on, 0, None) / SR / 0.02)
    k = int(auto.PRE_HOP * SR)
    flick = list(-6.0 + 8.0 * r.standard_normal(int(auto.ATTACK_S / auto.PRE_HOP)))   # a contour to follow
    pre = {'t_on': on / SR, 'db': [-80.0] * (on // k), 'attack_db': flick, 'v': 2}
    low = lambda v: _db(signal.sosfiltfilt(signal.butter(4, 150, 'lowpass', fs=SR, output='sos'), v))
    plain, hp = auto._pre_clamp(y, pre), auto._pre_clamp(y, dict(pre, hp_hz=auto.PRE_HP_HZ))
    assert low(plain) > low(y) + 10                               # the gain steps splatter into the empty sub band
    assert low(hp) < low(y) + 3                                   # the high-passed correction does not
    band = signal.sosfiltfilt(signal.butter(4, 3000, 'highpass', fs=SR, output='sos'), np.c_[plain, hp], axis=0)
    assert np.allclose(band[:, 0], band[:, 1], atol=0.02 * np.abs(band).max())   # the contour above it is the same


def test_noise_model_post_stage_keeps_the_clamp_only_when_it_helps():
    r = np.random.default_rng(7)
    n, on = int(0.12 * SR), int(0.02 * SR)
    env = (np.arange(n) >= on) * np.exp(-np.clip(np.arange(n) - on, 0, None) / SR / 0.03)
    x, y = (0.3 * r.standard_normal(n) * env for _ in range(2))  # a noise burst and a take of it: nothing to fix
    post = auto.fit_post(x, [y], noise=True)
    assert 'pre' not in post or post['pre'].get('hp_hz') == auto.PRE_HP_HZ
    assert 'hp_hz' not in (auto.fit_post(x, [y]).get('pre') or {})   # a modal / sines stage: as before
