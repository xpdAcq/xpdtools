#!/usr/bin/env python
import sys
from pathlib import Path
import pytest


if __name__ == "__main__":
    # show output results from every test function
    args = ["--showlocals"]
    # show the message output for skipped and expected failure tests
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    print("pytest arguments: {}".format(args))
    # # compute coverage stats for xpdAcq
    # args.extend(['--cov', 'xpdAcq'])
    # call pytest and exit with the return code from pytest so that
    # travis will fail correctly if tests fail
    exit_res = 0
    # the test filse are run separately to avoid FileNotFound Bug in
    for test_file in Path("xpdtools/tests").glob("test_*.py"):
        a = args.copy() + [str(test_file)]
        exit_res = pytest.main(a)
    sys.exit(exit_res)
