"""Film-level numbers from the earshot timelines of the reference films: loudness, cut rate, sync of cuts, tempo.

    python eval/forge/pro/film_table.py      # prints a Markdown table, writes out/refs_pro/films.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from videos import OUT, VIDEOS, WORK  # noqa: E402

rows = []
for ref in sorted(VIDEOS):
    r = json.load(open(f'{WORK}/tl/{ref}.json', encoding='utf-8'))
    st = r['sync']['cuts']['stats'] if r.get('sync', {}).get('cuts') else {}
    items = r['sync']['cuts']['items'] if r.get('sync', {}).get('cuts') else []
    offs = [i['offset_ms'] for i in items if i.get('offset_ms') is not None and i['status'] in ('OK', 'near')]
    m = r.get('music') or {}
    secs = ' / '.join(s.get('label', '')[:60] for s in r.get('sections', [])[:2])
    voice = any('voice' in (s.get('label') or '') for s in r.get('sections', []))
    row = {'film': ref, 'dur_s': round(r['duration'], 1), 'lufs': r['loudness'].get('integrated_lufs'),
           'tp_dbtp': r['loudness'].get('true_peak_dbtp'), 'lra': r['loudness'].get('lra_lu'),
           'cuts': len(r.get('cuts') or []), 'cuts_per_s': round(len(r.get('cuts') or []) / r['duration'], 2),
           'tempo': m.get('tempo'), 'pulse': m.get('pulse_clarity'), 'events_per_s': round(len(r['events']) / r['duration'], 1),
           'hit_1f': st.get('hit_within_1_frame'), 'chance_1f': st.get('chance_within_1_frame'),
           'hit_100ms': st.get('hit_within_100ms'), 'accent_1f': st.get('accent_within_1_frame'),
           'accent_chance_1f': st.get('accent_chance_within_1_frame'),
           'median_offset_ms': float(np.median(offs)) if offs else None,
           'early_share': round(float(np.mean([o < 0 for o in offs])), 2) if offs else None,
           'voice_over': voice, 'sections': secs}
    rows.append(row)
with open(f'{OUT}/films.json', 'w', encoding='utf-8') as fh:
    json.dump(rows, fh, indent=1)
print('| film | dur s | LUFS | LRA | cuts/s | tempo | VO | cut answered within 1 frame (chance) | strong accent within 1 frame (chance) | median signed offset ms (OK/near) |')
print('|---|---|---|---|---|---|---|---|---|---|')
for w in rows:
    print(f"| {w['film']} | {w['dur_s']} | {w['lufs']} | {w['lra']} | {w['cuts_per_s']} | "
          f"{round(w['tempo']) if w['tempo'] else '-'} (pulse {w['pulse']}) | {'yes' if w['voice_over'] else 'no'} | "
          f"{w['hit_1f']} ({w['chance_1f']}) | {w['accent_1f']} ({w['accent_chance_1f']}) | {w['median_offset_ms']} |")
