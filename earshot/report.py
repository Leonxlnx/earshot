"""Output helpers: every command returns a dict with a plain-language "summary" plus details.

Agents read the text form by default; --json prints the full dict; --out FILE also writes it to disk.
Conventions: times in seconds (3 decimals), levels in dBFS / LUFS / dBTP (1 decimal), similarities as cosine (3
decimals), calibrated scores as z (standard deviations above a background) or 0..1 probabilities.
"""
import json
import math
import os
import sys

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    return o


def to_json(result, indent=1):
    return json.dumps(_clean(result), indent=indent, ensure_ascii=False)


def render_text(result):
    """generic text rendering: the summary first, then compact key: value lines"""
    lines = []
    if result.get('summary'):
        lines.append(str(result['summary']))
    for k, v in result.items():
        if k in ('summary', 'text'):
            continue
        if isinstance(v, (dict, list)):
            s = json.dumps(_clean(v), ensure_ascii=False)
            if len(s) > 600:
                s = s[:600] + ' ...'
        else:
            s = str(v)
        lines.append(f'{k}: {s}')
    return '\n'.join(lines)


def emit(result, args, text=None):
    """print the result for a CLI call (text unless --json) and write --out if given"""
    if getattr(args, 'out', None):
        d = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(d, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(to_json(result))
    if getattr(args, 'json', False):
        # ASCII escapes: valid JSON whatever the console code page (a piped Windows stdout is often cp1252)
        sys.stdout.write(json.dumps(_clean(result), indent=1, ensure_ascii=True) + '\n')
    else:
        t = text if text is not None else (result.get('text') or render_text(result))
        try:
            sys.stdout.write(t + '\n')
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or 'ascii'
            sys.stdout.write(t.encode(enc, 'backslashreplace').decode(enc, 'replace') + '\n')
    return result


def error_line(cmd, msg):
    """the one-line error every command prints on stderr: 'earshot <cmd>: error: <msg>'"""
    msg = ' '.join(str(msg).split()) or 'unknown error'
    print(f'earshot {cmd}: error: {msg}', file=sys.stderr)


def fmt_t(t):
    """seconds -> m:ss.ss (agent-readable timestamps)"""
    if t is None:
        return '-'
    m = int(t // 60)
    return f'{m}:{t - 60 * m:05.2f}'
