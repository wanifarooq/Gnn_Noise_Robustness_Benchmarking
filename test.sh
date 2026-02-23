#!/usr/bin/env bash
#
# Smoke tests for all 13 training methods.
# Runs each method end-to-end on Cora (5 epochs, uniform noise 0.2).
#
# Usage:
#   ./test.sh                        # run all 13 methods
#   ./test.sh -k standard            # run a single method by name
#   ./test.sh -k "nrgnn or rtgnn"    # run several methods
#   ./test.sh -x                     # stop on first failure
#   ./test.sh --tb=long              # verbose tracebacks
#   ./test.sh -k erase --tb=short    # combine flags freely
#
# All arguments are forwarded to pytest.
#
set -euo pipefail
python -m pytest tests/test_smoke.py -v "$@"
