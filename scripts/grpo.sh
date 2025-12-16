#!/bin/bash

# shellcheck disable=SC2164
cd /home/zjx/ahua_cs336/assignment5-alignment

conda activate hua

uv run -m cs336_alignment.grpo

# nohup ./scripts/grpo.sh >> /home/zjx/ahua_cs336/assignment5-alignment/cs336_grpo.log 2>&1 &