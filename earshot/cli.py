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
    ap = argparse.ArgumentParser(prog='earshot', description='Ears for coding agents: describe, match, search, compare, '
                                 'time-align and critique sound files you cannot hear.')
    ap.add_argument('--threads', type=int, help='torch CPU threads (default: EARSHOT_THREADS or min(4, cores/2))')
    return ap


def _register(sub, modules):
    failed = {}
    done = set()
    for mod in modules:
        if mod in done:
            continue
        done.add(mod)
        try:
            importlib.import_module(f'earshot.{mod}').add_cli(sub)
        except Exception as e:  # a broken module must not break the others
            failed[mod] = f'{type(e).__name__}: {e}'
    return failed


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = _parser()
    sub = ap.add_subparsers(dest='command', metavar='<command>')
    cmd = next((a for a in argv if not a.startswith('-')), None)
    if cmd in COMMANDS:
        failed = _register(sub, [COMMANDS[cmd]])
        if failed:
            print(f'earshot: the "{cmd}" command is unavailable: {failed[COMMANDS[cmd]]}', file=sys.stderr)
            return 2
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
    try:
        return args.func(args) or 0
    except Exception as e:  # readable one-line errors for agents; EARSHOT_DEBUG=1 shows the traceback
        if os.environ.get('EARSHOT_DEBUG'):
            raise
        from .report import error_line
        error_line(args.command, f'{type(e).__name__}: {e}')
        return 1


def common(p):
    """flags shared by every command"""
    p.add_argument('--json', action='store_true', help='print the full result as JSON')
    p.add_argument('--out', help='also write the JSON result to this file')
    p.add_argument('--threads', type=int, default=argparse.SUPPRESS,
                   help='torch CPU threads (also accepted before the command; default EARSHOT_THREADS or min(4, cores/2))')
    return p


if __name__ == '__main__':
    sys.exit(main())
