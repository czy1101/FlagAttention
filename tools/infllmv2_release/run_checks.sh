#!/usr/bin/env bash
# Run from the FlagAttention repository root after selecting the experiment GPU.
set -euo pipefail
P="${PYTHON:-python}"
ROOT="$(pwd)"
QA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$ROOT/src/flag_attn/runtime/backend/_hygon/infllmv2/operator.py"
export FLAG_ATTN_BACKEND=hygon
export FLAG_ATTN_TEST_DEVICE="${FLAG_ATTN_TEST_DEVICE:-cuda:0}"
export PYTHONPATH="$ROOT/src:$ROOT:$QA${PYTHONPATH:+:$PYTHONPATH}"

# Original imports exercise common components as well as the public backend API.
"$P" -m pytest -q -ra tests/flag_attn/test_infllmv2_attention.py
# Bind component-level checks to Hygon without changing official assertions.
"$P" -m pytest -q -ra -p release_plugin tests/flag_attn/test_infllmv2_attention.py
"$P" -m pytest -q -s benchmark/test_infllmv2_attention_benchmark.py
"$P" "$QA/check_contract.py"
