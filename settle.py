#!/usr/bin/env python3
"""Backward-compat shim: `python settle.py --selftest` keeps working from a checkout.

The engine lives in settled_computer/engine.py.
"""
import sys

from settled_computer.engine import main

if __name__ == "__main__":
    sys.exit(main())
