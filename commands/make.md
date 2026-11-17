---
description: Make one sound with earshot from the words given and write it into the project
argument-hint: "[the sound in words, e.g. a soft success chime for a finance app, a wooden drawer closing]"
---

Make this sound with earshot: $ARGUMENTS

1. Pick the file: the folder this project keeps sounds in (for example `public/sounds/` or `assets/audio/`), else
   `sounds/`; a short name from the job (`success.mp3`); `.mp3` for a web project (`.webm` for a loop: an mp3 loop
   has a gap at its seam), `.wav` otherwise, unless the request names a format. If the request asks for variations
   ("three variations of a footstep"), keep the words: one call writes `name_1 .. name_N`. Keep the number of a
   thing a person does in a row as written too: "footsteps on gravel" or "knocking" is played as the action, "a
   footstep" or "a single knock" as one event.
2. Call the `make` tool of the earshot MCP server once, with `want` = the request as written and `out` = the absolute
   path. If no earshot MCP server is connected, run `earshot make "REQUEST" -o PATH` instead. No `full=true`,
   `--json` or `--listen`. If `earshot` is not found, say it must be installed first:
   `uv tool install "earshot[mcp] @ git+https://github.com/Leonxlnx/earshot"`.
3. Report the answer as given: the file line (design, length, pitch and voice for a tonal sound, loudness, `clean`
   or `audit: ...`), `also: ...`, a `played: ...` line (one event or the action), and any `not understood` /
   `not followed` line, and a line about the length (a cut, the shortest loop, whole bars, a counted action that
   could not fill it). A take with `audit:` is not usable: make it again with another `seed`. Do not say how it sounds beyond what the answer says; if the
   design is a learned one (a real thing such as a footstep, a switch, a door or a drum), say it is a new take of a
   public-domain recording, computed from a model of it (no audio file is shipped or played back).
