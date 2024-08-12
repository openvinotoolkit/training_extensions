#!.tox/fuzzing/bin/python

import sys
from pathlib import Path

from otx.cli import main as cli_main


def main():
    if len(sys.argv) != 2:
        print("usage: python eval_fuzzing_crash.py <fuzzing-crash-binary-file>")
        return

    ba = None
    if Path.exists(sys.argv[1]):
        with Path.open(sys.argv[1], "rb") as f:
            ba = bytearray(f.read())

    arguments = "".join(ba.decode(errors="replace"))

    sys.argv = ["otx", f"{arguments!s}"]
    _ = cli_main()


if __name__ == "__main__":
    main()
