---
description: Make one sound with earshot from the words given and write it into the project
argument-hint: "[the sound in words, e.g. a soft success chime for a finance app]"
---

Make this sound with earshot: $ARGUMENTS

1. Pick the file: the folder this project keeps sounds in (for example `public/sounds/` or `assets/audio/`), else
   `sounds/`; a short name from the job (`success.ogg`); `.ogg` for a web project, `.wav` otherwise, unless the
   request names a format.
2. Call the `make` tool of the earshot MCP server once, with `want` = the request as written and `out` = the absolute
   path. If no earshot MCP server is connected, run `earshot make "REQUEST" -o PATH` instead. No `full=true`,
   `--json` or `--listen`. If `earshot` is not found, say it must be installed first:
   `uv tool install "earshot[mcp] @ git+https://github.com/Leonxlnx/earshot"`.
3. Report the answer as given: the file line (design, length, pitch, voice, loudness, `clean` or `audit: ...`),
   `also: ...`, and any `not understood` / `not used by this sound` line. A take with `audit:` is not usable: make
   it again with another `seed`. Do not say how it sounds beyond what the answer says.
