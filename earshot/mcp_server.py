"""earshot as an MCP server: coding agents call earshot's ears as tools (stdio transport).

    earshot mcp              # started by the MCP client, not by hand; speaks JSON-RPC on stdin/stdout
    earshot mcp --warm       # also load the models in the background at startup (the first model call is then fast)
    earshot mcp --tools      # list the tools and exit
    python -m earshot.mcp_server [--warm]   # the same server without the CLI front end

Register with Claude Code (user scope = every project; -e passes environment variables to the server):
    claude mcp add earshot --scope user -e EARSHOT_THREADS=3 -- /path/to/earshot/.venv/bin/python -m earshot mcp
Any MCP client (project .mcp.json, Claude Desktop, Cursor ...):
    {"mcpServers": {"earshot": {"command": "/path/to/earshot/.venv/bin/python", "args": ["-m", "earshot", "mcp"],
                                "env": {"EARSHOT_THREADS": "3", "GEMINI_API_KEY": "${GEMINI_API_KEY:-}"}}}}
GEMINI_API_KEY is only for the optional llm_critique tool: keep the key in the user's own environment and reference it
as ${GEMINI_API_KEY:-} (Claude Code expands it at startup); never write a literal key into .mcp.json or any project
file. The server answers the handshake within a few seconds, longer on a cold disk or a busy machine; Claude Code waits
MCP_TIMEOUT ms for it, so on a slow machine start Claude Code with MCP_TIMEOUT=60000 (PowerShell:
$env:MCP_TIMEOUT=60000; claude) or the tools are missing for the whole session.

Tools: measure, describe, match, rank, find, compare, ab, realness, timeline, sheet, critique, doctor, llm_critique.
Models load on the first call that needs them (~20-60 s) and stay warm for the life of the server; calls run one at a
time. Each result is text: the plain-language summary first (plus the listening transcript for timeline / the
critique text for llm_critique / one row per file for realness on a folder), then compact JSON (always valid JSON;
full=true for everything). sheet also returns the picture as an image. A tool whose module is missing from this
install answers with a clear message instead of failing the server.

Sound bank tools (no models, pure DSP): sfx_list, sfx_render (one take aimed at the picture), kit_render (a film's
kit.json into takes + kit.lock.json); earshot.forge.sfx / earshot.forge.kit.
"""
from __future__ import annotations

import contextlib
import functools
import importlib.util
import inspect
import io
import json
import os
import sys
import threading
import time
from typing import Annotated

from . import audio, config
from .report import _clean, emit

INSTRUCTIONS = """earshot gives you ears: you cannot hear, so use these tools whenever you must judge, choose or check
sound (sound effects, music, ambience, a mix, the audio of a video). Use absolute paths.
Loops: choose an asset = rank (many files) -> find (best moment in the winner) -> describe (verify; + realness when
the user wants real or professionally produced sound).
Review a mix or film = timeline (what happens when, sync to cuts/cues) -> sheet (look at the picture) -> critique
(scorecard vs professional launch films). A/B a change = ab on before/after renders, then timeline on the new one.
measure = exact meters (LUFS, dBTP, noise floor), no models. llm_critique = optional Gemini opinion, uploads audio:
only with the user's explicit consent.
Reading numbers: z = how far a description stands out among the few hundred vocabulary descriptions scored on the
same sound (>= 3 strong, 2.5-3 plausible, 2-2.5 weak, < 2 no match). p_real and AudioSet values are probabilities;
p_vs_alts is an uncalibrated forced choice among the alternatives you give (sharp, only as good as those
alternatives); cosine is raw CLAP similarity (not a probability, only comparable for the same sound and query). Trust
measurements most, verdicts and z next, cosine least. If realness is unavailable, say real-vs-synthetic was not
checked; describe's AudioSet synth classes (Sine wave, Beep, Synthesizer) are weaker evidence. The first call that
needs models takes ~20-60 s; later calls are fast (models stay loaded)."""

INSTRUCTIONS += """
Sounds for a film (no models, fast): sfx_list (the shipped bank) -> sfx_render (one take aimed at a cue: dur, apex,
pitch) or kit_render (a film's kit.json -> takes + kit.lock.json with measured onset / peak / end per take)."""

IMG_INLINE_MAX = 1600        # px, long edge of the inline copy (sent as JPEG q88: ~1/3 of the PNG bytes)
JSON_MAX = 7000
SILENT_PEAK_DBFS = -70.0     # same near-silence line as describe
TEXT_MAX = 24000
_LOCK = threading.RLock()   # one tool call (or the warm-up) at a time: no import races, gentle on the CPU


class ToolFailure(Exception):
    """a readable error for the agent (becomes an MCP error result)"""


def _field_types():
    """tool signatures are string annotations resolved when the server starts: only `earshot mcp` imports pydantic"""
    from pydantic import Field
    globals().update(
        Field=Field,
        Path=Annotated[str, Field(description='absolute path to an audio or video file (wav, flac, ogg, mp3, m4a, or '
                                              'the audio track of mp4/mov/webm)')],
        Paths=Annotated[list[str] | str, Field(description='files, folders (searched recursively) or globs: one '
                                                           'string or a list')],
        Full=Annotated[bool, Field(description='return the complete JSON instead of the compact one')],
        Start=Annotated[float | None, Field(description='optional start of the segment to analyse, seconds')],
        End=Annotated[float | None, Field(description='optional end of the segment, seconds')],
        Query=Annotated[str, Field(description='a plain description of the sound, e.g. "a deep thocky mechanical '
                                               'keyboard key press" (describe the sound itself, not its use)')],
        Video=Annotated[str | None, Field(description='optional path to the film (mp4/mov) for picture cuts and '
                                                      'frames')],
        Cues=Annotated[list[float] | str | None, Field(description='optional picture event times: a list of seconds, '
                                                                   '"1.2,3.4", or an absolute path to a cues JSON '
                                                                   'file ({fps, events: [{f or t, kind}]})')])


def _file(path, what='file'):
    if not path or not isinstance(path, str):
        raise ToolFailure(f'{what}: a file path is required')
    p = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
    if not os.path.exists(p):
        raise ToolFailure(f'{what} not found: {p} (pass an absolute path; the server runs in {os.getcwd()})')
    if os.path.isdir(p):
        raise ToolFailure(f'{what} is a folder, expected a file: {p}')
    return p


def _opt_file(path, what):
    return _file(path, what) if path else None


def _many(paths, what='paths'):
    """files of every given file / folder / glob; a given path that yields nothing is an error, not silently skipped"""
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ToolFailure(f'{what}: give at least one file, folder or glob')
    expanded = [os.path.abspath(os.path.expanduser(str(p).strip().strip('"'))) for p in paths]
    empty = [p for p in expanded if not audio.list_media([p])]
    if empty:
        raise ToolFailure(f'{what}: no audio/video files found in {", ".join(empty[:5])} (no such file or folder, or '
                          f'nothing playable in it; the server runs in {os.getcwd()}). Fix or remove '
                          f'{"it" if len(empty) == 1 else "them"}.')
    return audio.list_media(expanded)


def _api(name):
    """the public earshot function `name`, imported lazily, or a ToolFailure saying it is not available yet"""
    from earshot import api
    try:
        return api._fn(name)
    except Exception as e:  # noqa: BLE001
        mod = api._FUNCS.get(name, (name, name))[0]
        if importlib.util.find_spec(f'earshot.{mod}') is None:
            present = [k for k in (f.__name__[5:] for f, _, _ in TOOLS)
                       if k not in api._FUNCS or importlib.util.find_spec(f'earshot.{api._FUNCS[k][0]}')]
            raise ToolFailure(f'earshot.{name} is not part of this install yet. Tools with their module present: '
                              f'{", ".join(present)}.') from None
        raise ToolFailure(f'earshot.{name} could not be loaded ({type(e).__name__}: {e}). If earshot is being '
                          'updated, retry in a moment; `earshot doctor` checks the install.') from None


def _call(fn, kw):
    """call fn with the non-None keyword arguments it accepts; note the ones this earshot version does not take"""
    kw = {k: v for k, v in kw.items() if v is not None}
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(**kw), []
    return fn(**{k: v for k, v in kw.items() if k in params}), [k for k in kw if k not in params]


def _shrink(o, n, depth=0):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, str):
        return o if len(o) <= 400 else o[:400] + ' ...'
    if isinstance(o, dict):
        return {k: _shrink(v, n, depth + 1) for k, v in o.items() if not (depth == 0 and k in ('summary', 'text'))}
    if isinstance(o, list):
        out = [_shrink(v, n, depth + 1) for v in o[:n]]
        return out + ([f'... +{len(o) - n} more'] if len(o) > n else [])
    return o


def _dumps(o):
    return json.dumps(o, ensure_ascii=False, separators=(',', ':'))


def compact(r, limit=JSON_MAX):
    """short, always valid JSON for an agent's context: floats to 3 decimals, long lists cut, summary/text left out;
    if that is still too long the largest top-level fields are left out and named in 'cut_fields'"""
    c = _clean(r)
    for n in (24, 12, 6, 3, 1):
        d = _shrink(c, n)
        s = _dumps(d)
        if len(s) <= limit:
            return s
    if not isinstance(d, dict):
        return _dumps({'cut': 'too long for the compact view; call again with full=true'})
    cut = []
    for k in sorted(d, key=lambda k: -len(_dumps(d[k]))):
        cut.append(k)
        s = _dumps({**{k2: v for k2, v in d.items() if k2 not in cut}, 'cut_fields': cut,
                    'note': 'fields left out to stay short; call again with full=true'})
        if len(s) <= limit:
            break
    return s


def _format(r, full=False, ignored=()):
    head = str(r.get('summary') or '').strip()
    body = str(r.get('text') or '').strip()
    if body.startswith(head):
        head = ''
    if len(body) > TEXT_MAX and not full:
        body = body[:TEXT_MAX] + f'\n... (text cut at {TEXT_MAX} characters; full=true gives all of it)'
    js = json.dumps(_clean(r), ensure_ascii=False) if full else compact(r)
    note = f'(this earshot version ignored: {", ".join(ignored)})' if ignored else ''
    return '\n\n'.join(x for x in (head, body, note, ('JSON: ' if full else 'JSON (compact): ') + js) if x)


def _result(name, kw, fn=None):
    """(result dict, ignored kwargs) of one earshot call; any failure becomes a readable ToolFailure"""
    with _LOCK, contextlib.redirect_stdout(sys.stderr):
        fn = fn or _api(name)
        try:
            r, ignored = _call(fn, kw)
        except ToolFailure:
            raise
        except Exception as e:  # noqa: BLE001
            raise ToolFailure(f'{name} failed: {type(e).__name__}: {e}') from None
    return (r if isinstance(r, dict) else {'summary': str(r)}), ignored


def _run(name, kw, full=False, fn=None):
    r, ignored = _result(name, kw, fn)
    return _format(r, full, ignored)


def _cues(cues):
    """cue times as given (list / "1.2,3.4" / JSON text) or an absolute path to a cues JSON file"""
    if isinstance(cues, str) and cues.strip() and cues.strip()[:1] not in '[{0123456789.-':
        return _file(cues, 'cues')
    return cues or None


def _detail(detail):
    if detail not in ('low', 'normal', 'high'):
        raise ToolFailure('detail must be "low", "normal" or "high"')


def measure_summary(m):
    """one plain sentence from a features.measure() dict"""
    if m.get('error'):
        return f'{m["duration"]:.3f} s: {m["error"]}.'
    if m.get('peak_dbfs', 0) < SILENT_PEAK_DBFS:
        what = 'digitally silent' if m.get('digital_silence_frac', 0) >= 0.999 else 'near-silent'
        return (f'{m["duration"]:.3f} s, {what} (peak {m["peak_dbfs"]:.0f} dBFS): nothing to measure; the loudness, '
                'spectrum and envelope values of silence are meaningless.')
    parts = [f'{m["duration"]:.3f} s']
    if 'lufs' in m:
        parts.append(f'{m["lufs"]} LUFS integrated' + (f' (short-term max {m["short_term_max_lufs"]})'
                                                       if 'short_term_max_lufs' in m else ''))
    else:
        parts.append(f'RMS {m["rms_dbfs"]} dBFS (too short for LUFS)')
    parts.append(f'true peak {m["true_peak_dbtp"]} dBTP' + (f', {m["clipped_samples"]} clipped samples'
                                                            if m.get('clipped_samples') else ''))
    if 'lra_lu' in m:
        parts.append(f'LRA {m["lra_lu"]} LU')
    parts.append(f'noise floor {m["noise_floor_dbfs"]} dBFS')
    if m.get('brightness_hz'):
        parts.append(f'brightness {m["brightness_hz"] / 1000:.1f} kHz')
    if m.get('bandwidth_hz'):
        parts.append(f'bandwidth {m["bandwidth_hz"] / 1000:.1f} kHz')
    if m.get('attack_ms') is not None:
        parts.append(f'attack {m["attack_ms"]:g} ms' + (f', decay {m["decay_ms"]:g} ms' if m.get('decay_ms') else ''))
    if 'onset_count' in m:
        parts.append(f'{m["onset_count"]} onsets')
    if 'tempo' in m:
        parts.append(f'tempo ~{m["tempo"]:g} BPM, key {m["key"][0]["key"] if m.get("key") else "?"}')
    s = ', '.join(parts) + '.'
    if m.get('digital_silence_frac', 0) > 0.01:
        s += (f' {100 * m["digital_silence_frac"]:.1f}% exact digital zeros (usually edit, fade or gate padding; weak '
              'evidence on its own: use realness to judge real vs synthesized).')
    return s


# ---------------------------------------------------------------------------------------------------------- tools
def tool_measure(path: Path, start: Start = None, end: End = None,
                 music: Annotated[bool, Field(description='also estimate tempo, beats and key (music only)')] = False,
                 light: Annotated[bool, Field(description='skip the slower parts (onsets, harmonic ratio, pitch): '
                                                          'levels and spectrum only')] = False,
                 full: Full = False) -> str:
    """Exact meter readings of one file, no AI models, fast: duration, integrated LUFS, true peak dBTP, loudness
    range, noise floor, clipped samples, exact digital zeros, attack/decay ms, brightness, bandwidth, band energy,
    onsets (music=true adds tempo, beats, key). Use for loudness / peak / clipping / silence checks, to compare levels
    before and after a mix change, or to find where events start. These numbers are measurements: trust them."""
    from . import features
    p = _file(path)

    def run(path, start=None, end=None, music=False, light=False):
        m = features.measure(path, music=music, start=start, end=end, light=light)
        return dict(summary=measure_summary(m), **m)
    return _run('measure', dict(path=p, start=start, end=end, music=music, light=light), full, run)


def tool_describe(path: Path, start: Start = None, end: End = None,
                  top: Annotated[int, Field(description='tags per category', ge=1, le=30)] = 8,
                  full: Full = False) -> str:
    """What does this sound like? Calibrated tags per category (source, material, action, character ...), AudioSet
    top labels and key measurements, with a one-paragraph summary. Use first on any unknown file, or to check that a
    downloaded asset really is what its title claims. Loads the CLAP + AST models (first call ~20-60 s)."""
    return _run('describe', dict(path=_file(path), start=start, end=end, top=top), full)


def tool_match(path: Path, query: Query,
               alts: Annotated[list[str] | None, Field(description='1-5 plausible alternative descriptions to '
                                                                   'compete with the query, e.g. ["a clicky blue '
                                                                   'switch", "a typewriter key"]')] = None,
               start: Start = None, end: End = None, full: Full = False) -> str:
    """How well one sound matches a text description, calibrated: z (how far the description stands out among the
    vocabulary descriptions scored on this sound; vocab_size gives their count), verdict (z >= 3 strong, 2.5-3
    plausible, 2-2.5 weak, below 2 no match), percentile, p_vs_alts (an uncalibrated, sharp forced choice of the query
    among your alternatives), raw cosine (not a probability) and the closest vocabulary descriptions (what else it
    could be). Use to verify a specific claim about one file ("is this a deep thocky keyboard press?"). Judge by z
    and verdict; p_vs_alts only says which of the given wordings fits best, so give plausible alternatives."""
    return _run('match', dict(path=_file(path), query=query, alts=alts, start=start, end=end), full)


def tool_rank(paths: Paths, query: Query,
              top: Annotated[int | None, Field(description='return only the best N', ge=1)] = 20,
              full: Full = False) -> str:
    """Sort many candidate files by how well they match a description (calibrated). Use to choose the best asset from
    a folder of downloads or a sample library; then run find on the winner for the exact moment, and describe +
    realness to verify it. Embeddings are cached, so re-ranking the same folder with a new query is fast."""
    return _run('rank', dict(paths=_many(paths), query=query, top=top), full)


def tool_find(path: Path, query: Query,
              top: Annotated[int, Field(description='number of segments', ge=1, le=50)] = 5,
              kind: Annotated[str, Field(description='"event" = onset-based trims only (hits / clicks / steps), '
                                                     '"texture" = plain windows only (beds, ambiences), "auto" reads '
                                                     'the query and scores both')] = 'auto',
              win: Annotated[float | None, Field(description='texture window length in seconds (default: duration / '
                                                             '6, clamped to 1-5 s)')] = None,
              hop: Annotated[float | None, Field(description='hop between texture windows in seconds')] = None,
              start: Start = None, end: End = None, full: Full = False) -> str:
    """Find the best moments inside one long recording that match a description: segments [{start, end, z, verdict
    kind, head_db, tail_db, cut ...}] in seconds. kind stroke / event = onset-based trims (5 ms before the attack,
    after the decay); window = a stretch whose edges are snapped to the sound for one-shot queries (plain, to be
    crossfaded, for beds). cut = 'head cut' / 'tail cut' when the trim starts or ends inside a sound (ranked lower:
    check before using it). Use when a sound pack, ambience bed or typing take holds many events and you need the
    single cleanest one; then describe / measure / realness / ab that segment with its start and end."""
    if kind not in ('auto', 'event', 'texture'):
        raise ToolFailure('kind must be "auto", "event" or "texture"')
    return _run('find', dict(path=_file(path), query=query, top=top, kind=kind, win=win, hop=hop, start=start,
                             end=end), full)


def tool_compare(path: Path,
                 refs: Annotated[list[str] | str, Field(description='reference files, folders or globs: one string '
                                                                    'or a list')],
                 k: Annotated[int, Field(description='nearest references to list', ge=1, le=50)] = 5,
                 start: Start = None, end: End = None, full: Full = False) -> str:
    """Similarity of one sound to a set of reference recordings: the nearest references, typicality (z >= -1 fits
    the set, z < -2 is an outlier) and which measurements differ from the references, in words. Use to check that a
    candidate fits an established palette, or sounds like a reference the user loves. start / end judge only that
    part of the file (e.g. a trim from find)."""
    return _run('compare', dict(path=_file(path), refs=_many(refs, 'refs'), k=k, start=start, end=end), full)


def tool_ab(a: Annotated[str, Field(description='path of sound A (e.g. the current version)')],
            b: Annotated[str, Field(description='path of sound B (e.g. the candidate / new render)')],
            query: Annotated[str | None, Field(description='optional description both should match')] = None,
            refs: Annotated[list[str] | str | None, Field(description='optional reference files, folders or globs '
                                                                      'both should resemble')] = None,
            a_start: Annotated[float | None, Field(description='use A only from this time (s), e.g. a find trim',
                                                   ge=0)] = None,
            a_end: Annotated[float | None, Field(description='use A only up to this time (s)', gt=0)] = None,
            b_start: Annotated[float | None, Field(description='use B only from this time (s)', ge=0)] = None,
            b_end: Annotated[float | None, Field(description='use B only up to this time (s)', gt=0)] = None,
            full: Full = False) -> str:
    """A/B two sounds: which is closer to a description and/or a reference set, and how they differ in measured
    terms (louder, brighter, longer decay ...). Use to decide between two candidates, or to confirm a mix change did
    what you intended (before vs after render). For multi-take files pass the trims (a_start / a_end, b_start /
    b_end; A and B may be the same file): a whole-file A/B compares every take plus the silence between them."""
    return _run('ab', dict(a=_file(a, 'a'), b=_file(b, 'b'), query=query,
                           refs=_many(refs, 'refs') if refs else None, a_start=a_start, a_end=a_end,
                           b_start=b_start, b_end=b_end), full)


def tool_realness(path: Annotated[str, Field(description='an audio/video file, or a folder / glob of candidates (one '
                                                   'row per file)')],
                  start: Start = None, end: End = None,
                  ast: Annotated[bool, Field(description='also run the AudioSet tagger for synth-class evidence '
                                                         '(slower)')] = False,
                  full: Full = False) -> str:
    """Does it sound like a real recording / professionally produced audio, or like simple code-synthesized sound
    (oscillators, filtered noise, simple envelopes, beeps)? p_real (a probability) with a verdict and the evidence in
    words; files over 6 s also get per-window verdicts; a folder or glob gives one row per file. Electronic music and
    pro sound design count as real: it flags simple procedural audio. Use when the user wants real or professionally
    produced sound (choosing recordings, checking a finished mix), or asks whether a sound reads as code-synthesized.
    Synthesized audio may be intended: report the verdict, and treat it as a defect only when the user asked for real
    sound. Strong evidence, not proof: read the evidence list."""
    fn = _api('realness')
    mod = sys.modules.get('earshot.realness')
    model = getattr(mod, 'MODEL_PATH', None)
    if model and not os.path.isfile(model):
        raise ToolFailure('realness is unavailable in this install (its model file is missing). Tell the user that '
                          'real-vs-synthesized was NOT checked; as weaker evidence use describe (AudioSet classes such '
                          'as Sine wave, Beep, Synthesizer) and measure.')
    p = os.path.abspath(os.path.expanduser(path.strip().strip('"'))) if isinstance(path, str) else ''
    if os.path.isfile(p):
        return _run('realness', dict(path=p, start=start, end=end, ast=ast or None), full, fn)
    if not os.path.isdir(p) and not any(ch in p for ch in '*?['):
        _file(path)   # a plain path that does not exist: the usual "not found" message
    files = _many(path)
    many = getattr(mod, 'realness_many', None)
    if many is None:
        raise ToolFailure('this earshot version judges one file per call: pass a file')
    r, ignored = _result('realness', dict(paths=files, ast=ast or None, start=start, end=end), many)
    rows = [r.get('summary', ''), 'p_real  verdict                  file: main reason']
    for f in r.get('files', []):
        pr = '-' if f.get('p_real') is None else f'{f["p_real"]:.2f}'
        rows.append(f'{pr:>6}  {f.get("verdict", ""):<24} {os.path.basename(f.get("path", ""))}: '
                    f'{f.get("why") or f.get("summary", "")}')
    r['text'] = '\n'.join(rows)
    return _format(r, full, ignored)


def tool_timeline(path: Path, video: Video = None,
                  cues: Cues = None, start: Start = None, end: End = None,
                  detail: Annotated[str, Field(description='"low" (fast), "normal" or "high" (every event tagged, '
                                                           'slow)')] = 'normal',
                  full: Full = False) -> str:
    """Listen through a whole soundtrack or film: a listening transcript (events with time, level, tags, AudioSet
    labels), loudness curve, picture cuts (with video) and sync stats (event-to-cut / event-to-cue offsets). Use to
    review a mix or a film's audio, to learn what happens when, and to check that sounds land on picture. For a video
    file the picture is read from the same file unless video= is given."""
    p = _file(path)
    _detail(detail)
    vid = _opt_file(video, 'video') or (p if os.path.splitext(p)[1].lower() in audio.VIDEO_EXTS else None)
    return _run('timeline', dict(path=p, video=vid, cues=_cues(cues), start=start, end=end, detail=detail), full)


def tool_sheet(path: Path, video: Video = None, cues: Cues = None, start: Start = None, end: End = None,
               detail: Annotated[str, Field(description='model budget of the underlying timeline: "low" (fast), '
                                                        '"normal" or "high" (every event tagged, slow)')] = 'normal',
               out_png: Annotated[str | None, Field(description='where to write the PNG (default: a sheets folder in '
                                                                'the earshot cache)')] = None,
               include_image: Annotated[bool, Field(description='also return the picture inline (needs a client '
                                                                'that shows images)')] = True,
               full: Full = False):
    """Draw an annotated picture of a soundtrack or film on one time axis: video frames, sections, labelled events, a
    spectrogram, waveform with loudness, beats, and cue markers coloured by sync (green = hit within a frame, amber =
    within 100 ms, red = none); cuts are dashed lines. Returns the image plus its PNG path. Use when you can look at
    images: structure, density, gaps and sync at a glance; zoom with start/end, confirm details with timeline."""
    p = _file(path)
    _detail(detail)
    vid = _opt_file(video, 'video') or (p if os.path.splitext(p)[1].lower() in audio.VIDEO_EXTS else None)
    if out_png:
        out = os.path.abspath(os.path.expanduser(out_png))
    else:
        span = '' if start is None and end is None else f'_{start or 0:g}-{"end" if end is None else f"{end:g}"}'
        out = os.path.join(config.HOME, 'sheets',
                           f'{os.path.splitext(os.path.basename(p))[0]}_{audio.file_id(p)[:8]}{span}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r, ignored = _result('sheet', dict(path=p, out_png=out, video=vid, cues=_cues(cues), start=start, end=end,
                                       detail=detail))
    png, text = r.get('png') or out, _format(r, full, ignored)
    if not os.path.isfile(png):
        return f'{text}\n\n(no PNG found at {png})'
    text = f'PNG: {png}\n\n{text}'
    return (text, _image_for_agent(png)) if include_image else text


def _image_for_agent(path):
    from PIL import Image
    im = Image.open(path)
    im.thumbnail((IMG_INLINE_MAX, IMG_INLINE_MAX))
    buf = io.BytesIO()
    im.convert('RGB').save(buf, 'JPEG', quality=88, optimize=True)
    return buf.getvalue()


def tool_critique(path: Path, video: Video = None,
                  profile: Annotated[str | None, Field(description='optional reference profile JSON from `earshot '
                                                                   'profile` (default: the bundled launch films)')]
                  = None,
                  fast: Annotated[bool, Field(description='skip the AudioSet and realness passes (seconds instead of '
                                                          'minutes)')] = False,
                  full: Full = False) -> str:
    """Launch-film sound scorecard: each metric of the soundtrack (loudness, true peak, dynamics, spectral balance,
    event density, sync to picture, realness ...) against the range measured on professional launch films, with
    status ok / low / high, weight, why it matters and concrete actions with timestamps, plus an overall verdict
    (launch-grade / close / needs work / not launch-grade). Use on a finished soundtrack or film before delivery, and
    again after each round of fixes."""
    p = _file(path)
    vid = _opt_file(video, 'video') or (p if os.path.splitext(p)[1].lower() in audio.VIDEO_EXTS else None)
    kw = dict(path=p, profile=_opt_file(profile, 'profile'), video=vid)
    if fast:
        kw.update(use_ast=False, use_realness=False)
    return _run('critique', kw, full)


def tool_doctor(load: Annotated[bool, Field(description='also load both models and time one window (~20-60 s)')]
                = False, full: Full = False) -> str:
    """Check the earshot install: python packages, ffmpeg, the models (pinned revision, what the cache holds, whether
    an offline flag is set), CPU threads, whether an LLM key is present. Use when a tool fails, or once before the
    first real use."""
    from . import doctor
    return _run('doctor', dict(load=load), full, doctor.check)


def tool_llm_critique(path: Path,
                      consent: Annotated[bool, Field(description='true ONLY if the user explicitly agreed that this '
                                                                 'audio is uploaded to Google')] = False,
                      prompt: Annotated[str | None, Field(description='optional specific question to answer as '
                                                                      'well')] = None,
                      video: Video = None,
                      with_timeline: Annotated[bool, Field(description="also send earshot's listening transcript as "
                                                                       'context (loads models)')] = False,
                      model: Annotated[str | None, Field(description='Gemini model id (default '
                                                                     'EARSHOT_GEMINI_MODEL or gemini-3.8-flash)')]
                      = None,
                      synth_intended: Annotated[bool, Field(description='true when the film uses synthesized or '
                                                                        'procedural sound on purpose: the critique '
                                                                        'then does not count synthesis as a flaw')]
                      = False,
                      start: Start = None, end: End = None, full: Full = False) -> str:
    """OPTIONAL second opinion from Google Gemini (an audio LLM): rubric critique of sync, realism, mix balance,
    loudness, cheap / synthetic moments, and concrete fixes with timestamps. Use after timeline / critique when a
    judgement in words would help (what sounds cheap, what a mixer would change). It UPLOADS the audio to Google: call
    with consent=true only after the user explicitly agreed, and it needs GEMINI_API_KEY in the server's environment.
    Without key or consent it returns how to enable it (nothing is sent). An opinion, not a measurement: timestamps
    are approximate; verify fixes with timeline / ab."""
    from . import llm
    p = _file(path)
    kw = dict(path=p, consent=consent, prompt=prompt, video=_opt_file(video, 'video'), with_timeline=with_timeline,
              model=model, start=start, end=end, synth_intended=synth_intended)
    return _run('llm_critique', kw, full, llm.critique)


TOOLS = [  # (function, read-only, talks to the internet)
    (tool_measure, True, False), (tool_describe, True, False), (tool_match, True, False), (tool_rank, True, False),
    (tool_find, True, False), (tool_compare, True, False), (tool_ab, True, False), (tool_realness, True, False),
    (tool_timeline, True, False), (tool_sheet, False, False), (tool_critique, True, False),
    (tool_doctor, True, False), (tool_llm_critique, False, True),   # uploads audio: clients should ask first
]


# ------------------------------------------------------------------------------------- the sound bank (no models)
def _out_path(path, what):
    if not path or not isinstance(path, str):
        raise ToolFailure(f'{what}: an absolute path is required')
    return os.path.abspath(os.path.expanduser(path.strip().strip('"')))


def tool_sfx_list(category: Annotated[str | None, Field(description='only this category, e.g. "whoosh_fast", '
                                                                    '"pro_impact", "typing_key", "pluck"')] = None,
                  kind: Annotated[str | None, Field(description='only sounds for this cue kind or kit role, e.g. '
                                                                '"whoosh" (short and long), "hit", "key", "bell"')]
                  = None,
                  full: Full = False) -> str:
    """List earshot's shipped sound bank (no models, instant): every id (<category>/<member>, pluck/<name>,
    fit/<role>/<name>) with its length, onset and peak time, the cue kinds it serves and its knobs (dur; apex for
    the whoosh family; pitch for tonal sounds, with the native note). Use to choose sounds for a film's cues or kit
    before sfx_render / kit_render; filter by category or kind to keep it short."""
    from .forge import sfx

    def run(category=None, kind=None):
        r = sfx.sfx_list(category, kind)
        lines = [r['summary']]
        for x in r['sounds']:
            kn = x['knobs']
            k = [f"dur {kn['dur']}"] + (['apex'] if kn['apex'] else []) + (
                [f"pitch {kn['pitch'].get('note')}"] if kn['pitch'] else [])
            lines.append(f"{x['id']}: {x['dur_s']:.3f} s, peak {x['peak_s']:.3f} s; {','.join(x['kinds']) or '-'}; "
                         f"{', '.join(k)}")
        r['text'] = '\n'.join(lines)
        return r
    return _run('sfx_list', dict(category=category, kind=kind), full, run)


def tool_sfx_render(sound: Annotated[str, Field(description='bank id from sfx_list, e.g. "whoosh_fast/crisp"')],
                    out: Annotated[str, Field(description='absolute path of the WAV to write (24-bit)')],
                    seed: Annotated[int | None, Field(description="noise seed for a new take (default: the sound's "
                                                                  'own, the delivered final)')] = None,
                    dur: Annotated[float | None, Field(description='length in seconds (exact)', gt=0)] = None,
                    apex: Annotated[float | None, Field(description='seconds from the start to the loudest moment '
                                                                    '(whoosh family)', gt=0)] = None,
                    pitch: Annotated[str | None, Field(description='note name ("E5", "Eb6") or Hz ("440"), through '
                                                                   "the recipe's own pitch parameter")] = None,
                    peak_db: Annotated[float, Field(description='sample peak in dBFS', le=0)] = -1.0,
                    full: Full = False) -> str:
    """Render one sound of earshot's shipped bank to a WAV (no models, about a second), aimed at the picture:
    exact dur, the loudest moment on apex (whooshes), a note through the recipe's own pitch parameter (never
    resampled). Returns the measured onset, peak and end (for placing it on a cue) and the model-free audit (clicks,
    truncation, clipping, ring, rising pitch glide). Use for a single cue or to audition a candidate before a kit."""
    from .forge import sfx
    p = _out_path(out, 'out')
    return _run('sfx_render', dict(sound=sound, out=p, seed=seed, dur=dur, apex=apex, pitch=pitch, peak_db=peak_db),
                full, sfx.sfx_render)


def tool_kit_render(kit: Annotated[str, Field(description="absolute path of the film's kit.json (earshot kit init "
                                                         'writes one from a preset: calm, crisp, minimal, launch)')],
                    out: Annotated[str, Field(description='absolute path of the folder for the takes and '
                                                          'kit.lock.json')],
                    force: Annotated[bool, Field(description='render every take again (default: keep the takes '
                                                             'whose cache key and file still match)')] = False,
                    full: Full = False) -> str:
    """Render a film's sound kit (no models): every role's takes from the bank into WAVs plus kit.lock.json with the
    measured onset, peak, end and loudness of each take, cached by sound, seed, edits and bank version, every take
    audited model-free (a take that fails is re-rolled on a new seed). Use when a film's cues need sounds: the lock
    is what a mixer or cinetic's score.py places on the cue frames."""
    from .forge import kit as _kit
    return _run('kit_render', dict(kit=_file(kit, 'kit'), out=_out_path(out, 'out'), force=force), full,
                _kit.kit_render)


TOOLS += [(tool_sfx_list, True, False), (tool_sfx_render, False, False), (tool_kit_render, False, False)]


def _sdk():
    try:
        from mcp.server.mcpserver import Image, MCPServer as Server
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server, Image
        from mcp.server.fastmcp.exceptions import ToolError
    from mcp.types import ToolAnnotations
    return Server, Image, ToolError, ToolAnnotations


def _adapt(fn, Image, ToolError):
    @functools.wraps(fn)
    def run(**kw):
        try:
            with _LOCK:
                out = fn(**kw)
        except ToolFailure as e:
            raise ToolError(str(e)) from None
        if isinstance(out, tuple):
            return [out[0], Image(data=out[1], format='jpeg')]
        return out
    return run


def build():
    """the configured MCP server object (not running)"""
    from . import __version__
    Server, Image, ToolError, ToolAnnotations = _sdk()
    _field_types()
    try:
        srv = Server(name='earshot', instructions=INSTRUCTIONS, log_level='WARNING', version=__version__)
    except TypeError:  # mcp 1.x FastMCP: no version argument
        srv = Server(name='earshot', instructions=INSTRUCTIONS, log_level='WARNING')  # stderr stays readable
        low = getattr(srv, '_mcp_server', None)
        if low is not None:
            low.version = __version__
    for fn, ro, net in TOOLS:
        name = fn.__name__[len('tool_'):]
        kw = dict(name=name, description=_doc(fn),
                  annotations=ToolAnnotations(title=f'earshot {name}', readOnlyHint=ro, destructiveHint=False,
                                              openWorldHint=net))
        try:
            srv.add_tool(_adapt(fn, Image, ToolError), structured_output=False, **kw)
        except TypeError:  # older SDKs without structured_output
            srv.add_tool(_adapt(fn, Image, ToolError), **kw)
    srv.earshot_version = __version__
    return srv


def warm():
    """load the models in a background thread, holding the tool lock (two threads importing torch / scipy for the first
    time at once can see half-initialised modules). No stdout redirect here: it raced with the stdio transport, which
    picks up sys.stdout when it starts (the initialize answer then went to stderr and the client hung)."""
    def load():
        t = time.time()
        with _LOCK:
            try:
                import numpy as np
                from . import features
                from .models import ast, clap
                clap.get()
                ast.get()
                features.measure(np.random.RandomState(0).randn(48000).astype(np.float32) * 0.1)  # librosa import + JIT
                if importlib.util.find_spec('earshot.vocab'):
                    from . import vocab
                    vocab.get()
                print(f'earshot mcp: models warm ({time.time() - t:.0f} s)', file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f'earshot mcp: warm-up failed: {type(e).__name__}: {e}', file=sys.stderr)
    threading.Thread(target=load, daemon=True).start()


def _doc(fn):
    return ' '.join(fn.__doc__.split())


def serve(args):
    if getattr(args, 'tools', False):
        rows = [{'name': f.__name__[5:], 'read_only': ro, 'network': net, 'description': _doc(f)}
                for f, ro, net in TOOLS]
        head = f'{len(rows)} MCP tools. Register: claude mcp add earshot -- {sys.executable} -m earshot mcp'
        lines = [f'{r["name"]:13s} {r["description"].split(". ")[0]}.' for r in rows]
        return emit({'summary': head, 'tools': rows}, args, text='\n'.join([head] + lines)) and 0
    try:
        srv = build()
    except ImportError as e:
        print(f'earshot mcp needs the MCP SDK: pip install "earshot[mcp]" ({e})', file=sys.stderr)
        return 1
    if args.warm:
        warm()
    print(f'earshot mcp {srv.earshot_version}: serving {len(TOOLS)} tools on stdio (threads={config.threads()})',
          file=sys.stderr)
    srv.run('stdio')
    return 0


def add_cli(sub):
    from .cli import common
    p = common(sub.add_parser('mcp', help='run earshot as an MCP server (stdio) for coding agents',
                              description='MCP stdio server. Register: claude mcp add earshot -- '
                                          '<python> -m earshot mcp'))
    p.add_argument('--warm', action='store_true', help='load CLAP + AST in the background at startup')
    p.add_argument('--tools', action='store_true', help='list the tools and exit (does not start the server)')
    p.set_defaults(func=serve)
    return p


if __name__ == '__main__':   # python -m earshot.mcp_server [--warm]: the same server without loading the other commands
    import argparse
    sys.exit(serve(add_cli(argparse.ArgumentParser(prog='earshot').add_subparsers()).parse_args()))
