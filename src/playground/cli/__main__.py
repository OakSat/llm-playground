"""Entry point for the `slm` command.

Subcommands (`models`, `extract`, `eval`) are implemented in phase 4; this
module currently only establishes the console-script wiring.
"""

import sys

from playground import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the `slm` CLI. Returns a process exit code."""
    args = sys.argv[1:] if argv is None else argv

    if args and args[0] in {"-V", "--version"}:
        print(f"slm {__version__}")
        return 0

    print(f"slm {__version__} - no subcommands are implemented yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
