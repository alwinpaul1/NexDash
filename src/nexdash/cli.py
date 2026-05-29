"""Command-line interface for the NexDash dispatcher agent.

Provides two modes:

* ``python -m nexdash.cli`` — interactive REPL where a fleet dispatcher can ask
  natural-language questions about eActros 600 range and energy.
* ``python -m nexdash.cli --once "<question>"`` — answer a single question and
  exit (handy for scripting / smoke tests).

The CLI delegates all reasoning to :class:`nexdash.agent.DispatcherAgent`, which
runs an Anthropic tool-use loop. ANSI colours are used for friendly terminal
output; if the ``ANTHROPIC_API_KEY`` environment variable is missing we print
clear setup guidance and exit non-zero instead of crashing with a stack trace.
"""

from __future__ import annotations

import argparse
import os
import sys

from nexdash.agent import DispatcherAgent

# --------------------------------------------------------------------------- #
# Terminal styling helpers
# --------------------------------------------------------------------------- #

# ANSI escape codes. Disabled automatically when stdout is not a TTY so piped
# output stays clean.
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap ``text`` in an ANSI ``code`` when colour output is enabled."""
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _c("1", text)


def _green(text: str) -> str:
    return _c("32", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _red(text: str) -> str:
    return _c("31", text)


def _dim(text: str) -> str:
    return _c("2", text)


BANNER = _green(
    "\n"
    "  ███╗   ██╗███████╗██╗  ██╗██████╗  █████╗ ███████╗██╗  ██╗\n"
    "  ████╗  ██║██╔════╝╚██╗██╔╝██╔══██╗██╔══██╗██╔════╝██║  ██║\n"
    "  ██╔██╗ ██║█████╗   ╚███╔╝ ██║  ██║███████║███████╗███████║\n"
    "  ██║╚██╗██║██╔══╝   ██╔██╗ ██║  ██║██╔══██║╚════██║██╔══██║\n"
    "  ██║ ╚████║███████╗██╔╝ ██╗██████╔╝██║  ██║███████║██║  ██║\n"
    "  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝\n"
)

WELCOME = (
    f"{BANNER}"
    f"  {_bold('eActros 600 Range Intelligence — Fleet Dispatcher Assistant')}\n"
    f"  {_dim('Ask about energy use and route reachability in plain language.')}\n"
    f"  {_dim('Type a question, or one of: ')}"
    f"{_cyan('help')}{_dim(', ')}{_cyan('exit')}{_dim('.')}\n"
)

HELP_TEXT = (
    f"\n{_bold('What can I ask?')}\n"
    f"  • {_dim('How much energy to drive 80 km with 18 t payload at 75 kph?')}\n"
    f"  • {_dim('At 60% SOC can I reach 240 km with 12 t over a 3% climb at -5C?')}\n"
    f"  • {_dim('Compare energy use empty vs fully loaded on a 50 km flat route.')}\n\n"
    f"{_bold('Commands')}\n"
    f"  {_cyan('help')}  {_dim('show this message')}\n"
    f"  {_cyan('exit')}  {_dim('quit (also: quit, q, Ctrl-D)')}\n"
)

_API_KEY_ENV = "ANTHROPIC_API_KEY"

MISSING_KEY_MESSAGE = (
    f"{_red('✗ Missing ANTHROPIC_API_KEY')}\n\n"
    "The dispatcher assistant talks to the Anthropic API and needs a key.\n\n"
    f"{_bold('Set it for this shell:')}\n"
    f"  {_cyan('export ANTHROPIC_API_KEY=sk-ant-...')}\n\n"
    f"{_bold('Or add it to a .env file')} "
    f"{_dim('(see .env.example in the repo root):')}\n"
    f"  {_cyan('cp .env.example .env')}\n"
    f"  {_dim('# then edit .env and fill in your key')}\n\n"
    f"{_dim('Get a key at https://console.anthropic.com/settings/keys')}\n"
)


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #

def _has_api_key() -> bool:
    """Return True when an Anthropic API key is present and non-empty."""
    return bool(os.environ.get(_API_KEY_ENV, "").strip())


def _answer_once(agent: DispatcherAgent, question: str) -> int:
    """Ask a single question, print the answer, and return an exit code."""
    question = question.strip()
    if not question:
        print(_yellow("No question provided."), file=sys.stderr)
        return 1
    try:
        answer = agent.ask(question)
    except Exception as exc:  # noqa: BLE001 - surface any agent/API error clearly
        print(_red(f"Error while answering: {exc}"), file=sys.stderr)
        return 1
    print(answer)
    return 0


def _repl(agent: DispatcherAgent) -> int:
    """Run the interactive read-eval-print loop. Returns a process exit code."""
    print(WELCOME)
    while True:
        try:
            raw = input(_green("dispatch ▸ "))
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C: exit gracefully on its own line.
            print()
            break

        question = raw.strip()
        if not question:
            continue

        lowered = question.lower()
        if lowered in {"exit", "quit", "q"}:
            break
        if lowered in {"help", "?"}:
            print(HELP_TEXT)
            continue

        try:
            answer = agent.ask(question)
        except KeyboardInterrupt:
            print(_yellow("\n(interrupted)"))
            continue
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on errors
            print(_red(f"Error: {exc}"), file=sys.stderr)
            continue

        print(f"\n{_cyan(answer)}\n")

    print(_dim("Safe travels. ⚡"))
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexdash.cli",
        description=(
            "NexDash fleet dispatcher assistant for the Mercedes-Benz "
            "eActros 600. Ask about energy use and route reachability."
        ),
    )
    parser.add_argument(
        "--once",
        metavar="QUESTION",
        help="Answer a single question and exit (non-interactive).",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-8",
        help="Anthropic model id to use (default: claude-opus-4-8).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 == success)."""
    args = _build_parser().parse_args(argv)

    if not _has_api_key():
        print(MISSING_KEY_MESSAGE, file=sys.stderr)
        return 2

    agent = DispatcherAgent(model=args.model)

    if args.once is not None:
        return _answer_once(agent, args.once)
    return _repl(agent)


if __name__ == "__main__":
    sys.exit(main())
