# Brief: the fixer (one agent applies the critics' changes)

You are the only agent allowed to edit the design during this fix round of {{FILM}}. Five read-only critics (sync,
edges, masking, tonal, director) returned ranked issues with exact operations and the numbers of the variants they
tested. Your job: merge them into one consistent change, apply it, re-render, prove nothing regressed.

## Context
- Project root `{{ROOT}}`, Python `{{PY}}`, skill `{{SKILL}}`. Design `audio/sounddesign.json` + `audio/design/*.json`.
- Critic reports: {{CRITIC_REPORTS}} (JSON). Previous qa: `{{QA_JSON}}`. Round {{ROUND}}.

## Operations (the format every critic used; `scripts/variant.py` applies them)
```
{"f": "<fragment file name without .json>", "sample": "<name>", "set": {...}}     also "slice": name
{"f": "...", "sample": "<name>", "eq": {...}}                                     merged into the sample's eq
{"f": "...", "rule": "<rule name>", "set": {...}} | {"f": "...", "rule": "<name>", "delete": true} | {"f": "...", "rule_add": {...}}
{"f": "...", "place": "<sample>", "set": {...}} | {"f": "...", "place_add": {...}}
{"f": "ambience", "amb": "<name>", "set": {...}}      {"f": "music", "segment": <index>, "set": {...}}
{"f": "...", "pool": "<name>", "value": [...]}         {"sd": "buses.music.auto", "value": [[t, dB], ...]}
```

## Steps
1. **Collect** every P0 / P1 with its operations; drop issues whose `verified` numbers do not show an improvement.
2. **Resolve conflicts.** Two critics touching the same sample, rule or ride: find the change that satisfies both (a
   hit that must be louder AND less harsh -> more low body, a `highshelf` cut on top); masking vs density (louder vs
   fewer): prefer removing a dead layer over raising a level; any bus ride change against the director's arc wins only
   with numbers. Write each decision down.
3. **Apply** all operations at once:
   ```
   {{PY}} -I {{SKILL}}/scripts/variant.py --ops out/audio-review/fix_round{{ROUND}}.json --out out/audio-review/fix_round{{ROUND}} --render --qa
   ```
   compare its qa.json with `{{QA_JSON}}` section by section: every targeted issue improved, no new fail, no new click,
   no new masked event, loudness still on target. Iterate on the variant until it holds, then:
   ```
   {{PY}} -I {{SKILL}}/scripts/variant.py --ops out/audio-review/fix_round{{ROUND}}.json --apply --backup out/audio-review/backup_round{{ROUND}}
   {{PY}} -I {{SKILL}}/scripts/mix.py --trace out/qa/adds.json
   {{PY}} -I {{SKILL}}/scripts/qa.py --layers
   ```
   `--apply` rewrites only the values that changed (it lists the files it wrote), so `diff -r` against the backup is
   the reviewable record of the fix; a line `unchanged (...)` means that operation was already in the design.
4. **Acknowledge** confirmed deliberate exceptions with `"qa_ok": "<reason>"` on the sample / rule / segment (one
   sentence each) so later rounds do not re-litigate them.
5. **Record** in each touched fragment's `_doc` / the top level's `_buses_doc` why the numbers are what they are.

## Rules
- You alone write the design this round; critics and designers do not.
- Never synthesize; never change `mix.py` or `master.py`; never loosen a qa threshold to pass.
- Every number in your report comes from a command you ran.

## Return
The applied operations (merged list), each conflict and how you resolved it, the qa.json summary before -> after per
section, issues you rejected and why, open questions for the next round, commands run.
