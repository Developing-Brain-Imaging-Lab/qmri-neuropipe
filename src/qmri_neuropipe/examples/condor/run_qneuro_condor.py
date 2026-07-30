#!/usr/bin/env python3
"""Compatibility entry point for the canonical qneuro Condor worker."""

from __future__ import annotations

import sys

try:
    from qmri_neuropipe.examples.condor.qneuro_condor import cmd_run
except ImportError:  # Support an unpacked source checkout.
    from qneuro_condor import cmd_run  # type: ignore[no-redef]


def main(argv: list[str]) -> int:
    return cmd_run(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
