"""CLI entry point — routes to extract or translate subcommands."""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'extract':
        from .commands.extract_cmd import run

        run(sys.argv[2:])
    else:
        from .commands.translate_cmd import run

        run()


if __name__ == '__main__':
    main()
