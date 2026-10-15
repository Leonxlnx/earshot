"""Sound recipes for earshot forge: one module per sound type (SPACE, DEFAULTS, TARGET, ALTS, render)."""
import os

# where the recipes' default real reference recordings live (critic references only; pass --refs to use your own)
REF_ROOT = os.environ.get('EARSHOT_REFS', 'refs').replace(chr(92), '/').rstrip('/')
