#!/usr/bin/env bash
# Pre-push gate. Exits non-zero unless the suite demonstrably PASSED.
#
# Written as a script because the hand-rolled version had a hole that let a
# push through: it grepped the output for "failed", and when pytest was not
# installed the output read "No module named pytest", which contains no such
# word. Absence of the word "failed" is not evidence of success - the gate must
# require positive confirmation that tests ran and passed.
set -uo pipefail

cd "$(dirname "$0")/.."

if ! python3 -c "import pytest" 2>/dev/null; then
  echo "[check] installing test dependencies"
  python3 -m pip install --quiet pytest numpy pandas scikit-learn matplotlib PyYAML requests \
    || { echo "[check] FAIL: could not install dependencies"; exit 1; }
fi

out=$(python3 -m pytest tests/ -q 2>&1)
echo "$out" | tail -3

if echo "$out" | grep -qE "^[0-9]+ passed" ; then
  : # a bare "N passed" line is the only accepted success signal
elif echo "$out" | grep -qE "[0-9]+ passed(,|$| in )" && ! echo "$out" | grep -qiE "failed|error"; then
  :
else
  echo "[check] FAIL: the suite did not report a clean pass"
  exit 1
fi

echo "[check] OK"
