"""MediQuest Triage Desk — SATS-cited triage priorities for an outpatient triage desk. The agent proposes; a named
nurse decides."""


def main() -> None:
    from .cli import main as cli_main

    cli_main()
