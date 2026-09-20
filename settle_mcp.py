#!/usr/bin/env python3
"""Backward-compat shim: run the settled_computer MCP server from a source checkout.

The package is the real code (settled_computer/server.py). This file exists so
MCP configs registered before the 0.1 packaging keep working unchanged.
Prefer the installed console script once installed:

    pip install "settled-computer[desktop]"
    settled-computer          # stdio MCP server
    settled-computer --check  # verify capture + coordinates
"""
from settled_computer.server import main

if __name__ == "__main__":
    main()
