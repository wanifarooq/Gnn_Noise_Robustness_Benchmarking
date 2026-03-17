#!/usr/bin/env bash
#
# Run the full test suite (99 tests across 13 methods).
#
# Usage:
#   ./test.sh                        # run all tests
#   ./test.sh -k standard            # run a single method by name
#   ./test.sh -k "nrgnn or rtgnn"    # run several methods
#   ./test.sh -x                     # stop on first failure
#   ./test.sh --tb=long              # verbose tracebacks
#   ./test.sh -k erase --tb=short    # combine flags freely
#
# All arguments are forwarded to pytest.
#
set -euo pipefail
python -m pytest tests/ -v "$@"
