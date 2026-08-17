"""Stdout tee used by the long-running training scripts.

Was copied verbatim into three scripts before it lived here."""

import sys

class DualLogger:
    """Intercepts sys.stdout to print to both the terminal and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
