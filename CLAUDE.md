# CS336: Language Modeling from Scratch (Stanford Spring 2025)

Personal solutions to the Stanford CS336 course assignments by aHuaXu.

## Repository Structure

This is **not** a monorepo — each assignment is an **independent project** with its own git repo, `.venv`, `pyproject.toml`, and `uv.lock`. The top-level directory is simply a parent folder grouping them together.

```
cs336/
├── assignment1-basics/      # Assignment 1: Basics (LM from scratch)
│   ├── cs336_basics/        # Source: Transformer, BPE tokenizer, AdamW, training
│   └── tests/               # pytest tests with adapters.py bridge pattern
├── cs336_assignment2/       # Assignment 2: Systems (distributed training)
│   ├── cs336-basics/        # Staff reference implementation of assignment 1
│   ├── cs336_systems/       # Source: DDP, all-reduce, bucketed gradient sync
│   └── tests/               # pytest tests for DDP, attention, sharded optimizer
├── assignment3-scaling/     # Assignment 3: Scaling Laws
│   ├── cs336_scaling/       # Source: IsoFLOP analysis, power-law fitting
│   └── data/                # isoflops_curves.json
├── assignment4-data/        # Assignment 4: Data Quality & Filtering
│   ├── cs336-basics/        # Staff reference implementation
│   ├── cs336_data/          # Source: data filtering/processing (mostly TODO)
│   └── tests/               # Tests for dedup, extract, langid, PII, quality, toxicity
├── assignment5-alignment/   # Assignment 5: Alignment (SFT + GRPO)
│   ├── cs336_alignment/     # Source: SFT, GRPO, vLLM wrapper, evaluation
│   ├── scripts/             # alpaca eval, safety evaluation
│   ├── data/                # models (Qwen2.5-Math-1.5B), MMLU, GSM8K
│   └── tests/               # Tests for SFT, GRPO, DPO, metrics
├── data/                    # Shared data directory (empty/gitignored)
├── type_check_example.py    # Standalone typing example
└── user_info_sync_flow.puml # PlantUML diagram
```

## Language & Runtime

- **Language**: Python (3.11+ generally, 3.12 for assignments 3 and 5)
- **Deep Learning Framework**: PyTorch (~2.6.0, ~2.7.0 depending on assignment)
- **Key Libraries**: einops, jaxtyping, tiktoken, wandb, transformers, vLLM (0.7.2), accelerate, flash-attn
- **Package Manager**: [uv](https://github.com/astral-sh/uv) — all assignments use `uv` for dependency management
- **Build Backend**: `uv_build` (assignments 1, 2) or setuptools (assignments 3, 4, 5)

## Build & Run Commands

Each assignment has its own virtual environment. Always `cd` into the assignment directory first.

```sh
# Install dependencies (automatic with uv run, or explicit):
uv sync

# Run any script:
uv run <python_file_or_module>

# Run tests:
uv run pytest

# Run tests with verbose output + JUnit XML (as used by submission scripts):
uv run pytest -v ./tests --junitxml=test_results.xml

# Run a specific module (e.g., training):
uv run -m cs336_basics.train_script         # Assignment 1
uv run -m cs336_systems.naive_ddp           # Assignment 2

# Build submission:
./make_submission.sh                         # Assignment 1
./test_and_make_submission.sh                # Assignments 2, 4, 5
```

### Assignment 5 Special Setup
```sh
# flash-attn requires two-step install:
uv sync --no-install-package flash-attn
uv sync
```

## Testing

- **Framework**: pytest
- **Test pattern**: `tests/adapters.py` acts as a bridge between tests and implementation — tests import from adapters, which import from the actual module. Complete `adapters.py` to connect your implementation.
- **Config** (in `pyproject.toml`):
  - `log_cli = true`, `log_cli_level = "WARNING"`
  - `addopts = "-s"` (no capture, so print statements are visible)
- **Snapshot tests**: `tests/_snapshots/` and `tests/fixtures/` (gitignored, downloaded separately)
- **Run a single test file**:
  ```sh
  uv run pytest tests/test_model.py -v
  ```

## Linting & Formatting

- **Linter/Formatter**: [Ruff](https://docs.astral.sh/ruff/)
- **Config** (in `pyproject.toml` for assignments 1, 2, 4):
  ```toml
  [tool.ruff]
  line-length = 120

  [tool.ruff.lint]
  extend-select = ["UP"]    # pyupgrade rules
  ignore = ["F722"]         # jaxtyping string annotations

  [tool.ruff.lint.extend-per-file-ignores]
  "__init__.py" = ["E402", "F401", "F403", "E501"]
  ```
- **Type checking**: `ty` (uv-native type checker, added as dev dependency in assignments 1, 2)

## Architecture & Design Patterns

### Assignment 1 — Basics (from-scratch Transformer LM)
- **Custom nn.Module implementations**: `LinearLayer`, `EmbeddingLayer`, `RMSNormLayer`, `SwiGLU`, `RotaryPositionalEmbedding`, `MultiHeadAttention` (no `nn.Linear` or `nn.Embedding`)
- **Custom optimizer**: `AdamW` and `SGD` extending `torch.optim.Optimizer`
- **BPE tokenizer**: trained from scratch with `train_bpe.py`, used via `BpeTokenizer` class
- **Training**: config via YAML (`config.yaml`), logging to wandb, memmap data loading
- **Data**: TinyStories / OpenWebText subsets, pre-tokenized to `.bin` memmap files

### Assignment 2 — Systems (distributed training)
- **Distributed primitives**: `torch.distributed` (gloo/nccl backends)
- **DDP implementations**: naive → flattened → overlap individual params → bucketed
- **Pattern**: `mp.spawn()` for multi-process training, explicit all-reduce gradient sync
- **Dependencies**: imports `cs336_basics` as an editable local package via `uv.sources`

### Assignment 3 — Scaling Laws
- **IsoFLOP analysis**: `scipy.optimize.curve_fit` for power-law fitting
- **Staff model**: `BasicsTransformerLM` in `cs336_scaling/model.py` (complete reference implementation)

### Assignment 4 — Data Quality
- **Web data processing**: WARC/WET parsing (fastwarc, resiliparse)
- **Quality filters**: language ID (fasttext), deduplication (MinHash via mmh3), PII detection, toxicity filtering
- **Staff basics**: includes `cs336-basics` as a local editable dependency for training

### Assignment 5 — Alignment (SFT + GRPO)
- **Model**: Qwen2.5-Math-1.5B (downloaded to `data/models/`)
- **Inference**: vLLM wrapper for fast generation
- **Training**: HuggingFace Transformers + manual training loops
- **SFT**: Supervised fine-tuning with rollout-based evaluation
- **GRPO**: Group Relative Policy Optimization with reward functions (DeepSeek R1-Zero style)
- **Evaluation**: GSM8K math tasks, MMLU, AlpacaEval
- **GPU config**: hardcoded `CUDA_VISIBLE_DEVICES = "6, 7"` in `init.py` (adjust for your setup)

## Git Remotes

Each assignment has its own GitHub repo under `aHuaXu`:
- `assignment1-basics` → `git@github.com:aHuaXu/stanford_cs336.git`
- `cs336_assignment2` → `git@github.com:aHuaXu/cs336_assignment2.git`
- `assignment5-alignment` → `git@github.com:aHuaXu/cs336_assignment5.git`

All on `main` branch.

## Code Conventions

- Comments and docstrings mix English and Chinese (中文注释 describing implementation details)
- Type hints used extensively: `jaxtyping` for tensor shapes, standard Python typing elsewhere
- `bfloat16` precision for training
- wandb used for experiment tracking across all assignments
- Config via YAML (assignment 1) or constructor args (assignments 2, 5)

## Data Downloads

Assignment 1 requires manual data download:
```sh
cd assignment1-basics/data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

Assignment 4 has a `get_assets.sh` script for downloading test fixtures.

Assignment 5 uses Qwen2.5-Math-1.5B from HuggingFace (stored in `data/models/`).
