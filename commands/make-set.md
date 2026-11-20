---
description: Make a whole set of sounds with earshot as one family (one voice, key and room, levels balanced) and write it into the project
argument-hint: "[ui|game|arcade|film or the product's sounds in words] [theme]"
---

Make a set of sounds with earshot: $ARGUMENTS

1. Read the words: a ready-made set (`ui` 17 interface sounds, each a normal thing a hand does: a key, a page, a
   drawer, knocks, a camera shutter, no instrument; `game` 22; `arcade` 16 chip sounds; `film` 17 motion-design sounds), a
   theme (soft, glass, wood, felt, tine, metal, electric, digital, retro; it sets the voice, key and room of the
   tonal sounds, and turns `ui` into its tonal figures: name one only when the words ask for that), and any sounds
   of their own (each a `name` for the file and a `want` in words, or a `sound` id). Pick the folder this project
   keeps sounds in, else `sounds/`; format `webm,mp3` for a web project (every browser plays one of the two), `wav`
   otherwise, unless the words name one.
2. Call the `make_set` tool of the earshot MCP server once, with `out` = the absolute folder, `set`, `theme`,
   `format` and, for sounds of their own, `sounds: [{"name": "tap", "want": "a soft tap"}, ...]`. If no earshot MCP
   server is connected, run `earshot make --set NAME --theme THEME -o FOLDER --format FMT`, or write the list as a
   JSON spec (`{"set": ..., "sounds": [...]}`) and run `earshot make --spec FILE -o FOLDER`. One call for the whole
   set, never one `make` per sound. If `earshot` is not found, say it must be installed first:
   `uv tool install "earshot[mcp] @ git+https://github.com/Leonxlnx/earshot"`.
3. Report the answer as given: the folder and count, one line per file, and any file whose line does not say
   `clean` (not usable: make that sound again with another `seed`). A line `another recording of ...`
   says two names play two recordings of one learned design, as intended; a line `same design as ...` means two
   names got one sound: ask for another with `sound`. The folder also holds `manifest.json`.
