"""earshot command line. Every subcommand lives in its own module, which exposes add_cli(subparsers).

    earshot [--threads N] <command> [args] [--json] [--out result.json] [--threads N]

Run `earshot -h` for the list and `earshot <command> -h` for details. Only the module of the requested command is
imported (fast start); `-h` imports them all to list them. Errors: one line on stderr, 'earshot <command>: error: ...',
exit 1 (with --json, describe / match / rank / find / compare / ab also print it as JSON on stdout).
"""
import argparse
import contextlib
import importlib
import os
import sys

# command -> module (order = order in --help)
COMMANDS = {
    'make': 'make',              # make a sound from a request in words, or a whole set from a spec
    'sounds': 'make',            # the catalogue of sounds make can make
    'recipe': 'recipe',          # sounds the catalogue lacks, built from its parts: the form, the cookbook
    'build': 'build',            # build a sound the catalogue lacks: draft, render, hear, tune
    'pack': 'pack',              # many sounds in one file with a map (an audio sprite)
    'describe': 'describe',      # what does this sound like? tags + measurements
    'match': 'match',            # how well does a sound match a text description (calibrated)
    'rank': 'search',            # rank many files by a description
    'find': 'search',            # find the best moments inside a long recording (sample trims)
    'compare': 'compare',        # similarity to reference recordings
    'ab': 'compare',             # A/B two sounds
    'realness': 'realness',      # real recording vs code-synthesized sound
    'timeline': 'timeline',      # listen through a whole soundtrack or film: events, tags, sync, loudness
    'sheet': 'sheet',            # annotated picture of a soundtrack (spectrogram + events + frames) for agents with eyes
    'critique': 'critique',      # launch-video sound scorecard against a profile of professional references
    'profile': 'profile',        # build a reference profile from your own reference films
    'forge': 'forge',            # make sounds in code and improve them by listening
    'fit': 'forge.fit',          # translate a recording into code (analysis by synthesis) and vary it
    'sfx': 'forge.sfx',          # one sound of the shipped bank by id, aimed at the picture; audit any sound effect
    'kit': 'forge.kit',          # a film's sound kit: one bank sound per role, rendered to takes with sync points
    'gallery': 'gallery',        # one static HTML page of forge results: players, spectrograms, numbers
    'measure': 'features',       # objective measurements only (no models)
    'llm': 'llm',                # optional audio-LLM critique (only with your own API key + consent)
    'mcp': 'mcp_server',         # run earshot as an MCP server for coding agents
    'doctor': 'doctor',          # check the install, models and CPU settings
}


def _parser():
    ap = argparse.ArgumentParser(prog='earshot', description='earshot makes sound effects from a request in words '
                                 '(make, sounds, recipe, pack; no model, no network) and listens to sound files for '
                                 'you: describe, match, rank, find, compare, realness, timeline, sheet, critique and '
                                 'measure (the listening commands need the listen extra).')
    ap.add_argument('--threads', type=int, help='CPU threads for torch and BLAS (default: EARSHOT_THREADS or '
                                                'min(4, cores/2))')
    ap.add_argument('--version', action='version', version=_version())
    return ap


def _version():
    from . import __version__
    return f'earshot {__version__}'


def _hold_blas(argv):
    """numpy and scipy each bundle an OpenBLAS that starts a worker thread per logical CPU and, on Windows, commits a
    buffer for each when first imported: ~0.4 GB per process on 12 CPUs before any audio is read, ~0.9 GB once a tool
    has loaded scipy, for no speed-up (the heavy work is torch, already held to EARSHOT_THREADS). Give BLAS the same
    limit, from --threads or EARSHOT_THREADS, unless OPENBLAS_NUM_THREADS is set. It only counts before numpy is
    first imported, so main() calls this before any command module loads."""
    if 'OPENBLAS_NUM_THREADS' in os.environ or 'numpy' in sys.modules:
        return
    n = None
    for i, a in enumerate(argv):  # the last one wins, as in argparse
        if a == '--threads' and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith('--threads='):
            n = a.split('=', 1)[1]
    from . import config
    try:
        n = int(n or 0)
        n = n if n > 0 else config.threads()   # --threads 0 means the default, as in main()
    except ValueError:  # a bad value: argparse (or the first model load) reports it
        return
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)


def _register(sub, modules, absent=None):
    """add the commands of `modules`; return {module: why} for a module that is broken. A module that only lacks an
    extra (a base install) is no fault: its commands are listed with a note and, when called, say what to install;
    `absent` collects {module: the plain sentence}"""
    failed = {}
    done = set()
    for mod in modules:
        if mod in done:
            continue
        done.add(mod)
        try:
            importlib.import_module(f'earshot.{mod}').add_cli(sub)
        except Exception as e:  # a broken module must not break the others
            from . import extras
            said = extras.plain(e)
            if said is None:
                failed[mod] = f'{type(e).__name__}: {e}'
                continue
            if absent is not None:
                absent[mod] = said
            for name in (c for c, m in COMMANDS.items() if m == mod and c not in sub.choices):
                extra = extras.NEEDS.get(name) or extras.EXTRA.get(getattr(e, 'name', None) or '', ('listen',))[0]
                p = sub.add_parser(name, help=f'(needs the {extra} extra)', add_help=False)
                p.add_argument('rest', nargs='*', help=argparse.SUPPRESS)
                p.set_defaults(func=lambda a, said=said: _absent(said))
    return failed


def _absent(said):
    from .extras import MissingExtra
    raise MissingExtra(said)


def _missing_line(cmd, said):
    """the one-line error of a command whose module needs an extra that is not installed"""
    from . import extras
    from .report import error_line
    try:
        extras.check(cmd)
    except extras.MissingExtra as m:
        said = str(m)
    error_line(cmd, said)
    return 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    _hold_blas(argv)
    ap = _parser()
    sub = ap.add_subparsers(dest='command', metavar='<command>')
    cmd = next((a for a in argv if not a.startswith('-')), None)
    if '--version' in (argv[:argv.index(cmd)] if cmd in argv else argv):   # no command module needs to load
        print(_version())
        return 0
    if cmd in COMMANDS:
        absent = {}
        failed = _register(sub, [COMMANDS[cmd]], absent)
        if failed:
            print(f'earshot: the "{cmd}" command is unavailable: {failed[COMMANDS[cmd]]}', file=sys.stderr)
            return 2
        if absent:
            return _missing_line(cmd, absent[COMMANDS[cmd]])
    else:
        failed = _register(sub, list(COMMANDS.values()))
        for m, e in failed.items():
            print(f'(earshot: module {m} unavailable: {e})', file=sys.stderr)
    args = ap.parse_args(argv)
    if args.threads:
        os.environ['EARSHOT_THREADS'] = str(args.threads)
    if not getattr(args, 'func', None):
        ap.print_help()
        return 0
    if args.command != 'mcp':  # a file name outside the console code page must never crash a print
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(errors='backslashreplace')
    from . import extras
    try:
        extras.check(args.command)      # a command whose extra is missing says so before it starts
        return args.func(args) or 0
    except Exception as e:  # readable one-line errors for agents; EARSHOT_DEBUG=1 shows the traceback
        if os.environ.get('EARSHOT_DEBUG'):
            raise
        from .report import error_line
        error_line(args.command, extras.plain(e, args.command) or f'{type(e).__name__}: {e}')
        return 1


def common(p):
    """flags shared by every command"""
    p.add_argument('--json', action='store_true', help='print the full result as JSON')
    p.add_argument('--out', help='also write the JSON result to this file')
    p.add_argument('--threads', type=int, default=argparse.SUPPRESS,
                   help='CPU threads for torch and BLAS (also accepted before the command; default EARSHOT_THREADS or '
                        'min(4, cores/2))')
    return p


if __name__ == '__main__':
    sys.exit(main())
