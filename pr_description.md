## What

Adds built-in support for `intfloat/multilingual-e5-small` as a dense pooled text embedding model.

Closes #123

## Why

This model was explicitly requested in #123. It is a small (~118M params), fast,
multilingual E5-family model, complementing the already-supported
`intfloat/multilingual-e5-large`, for use cases needing lower latency/memory footprint.

## How I validated correctness

- Canonical vector values were obtained from the reference HuggingFace `transformers`
  implementation of `intfloat/multilingual-e5-small` (mean pooling over the last hidden
  state, matching `PooledEmbedding` post-processing used for the whole E5 family),
  using the same input text (`"hello world"`) and comparison tolerance (`atol=1e-3`) as
  existing tests in `tests/test_text_onnx_embeddings.py`.
- The reference run was executed twice and produced identical values.
- Local fastembed output matches the reference with max abs diff of ~3.2e-7 (tolerance is 1e-3).
- `dim=384` and `size_in_GB=0.44` verified against the actually downloaded ONNX artifact
  (`onnx/model.onnx` = 470,268,510 bytes = 0.438 GiB).
- `tests/test_custom_models.py` previously used `intfloat/multilingual-e5-small` as the
  example custom model; since the model is now built-in, `add_custom_model` correctly
  rejects it. The example was switched to `Xenova/multilingual-e5-small` (same weights,
  so the existing canonical values in that test remain valid and the test passes).
  Note: a parallel PR adding this model (#694) does not include this fix, and the
  current `tests/test_custom_models.py` fails for any PR that registers this model
  as built-in without it.
- Ran the full `tests/test_text_onnx_embeddings.py` and `tests/test_custom_models.py`
  suites locally — all passing.
- Ran `ruff check` / `ruff format --check` / pre-commit hooks — all passing.

## Note on the parallel PR

While working on this, I noticed #694 addresses the same issue. This PR is submitted
independently; happy to consolidate with the author/maintainers on whichever version
is preferred.

## Checklist

- [x] Added model to `supported_pooled_models` in `fastembed/text/pooled_embedding.py`
- [x] Added canonical vector test in `tests/test_text_onnx_embeddings.py`
- [x] Fixed custom-model example in `tests/test_custom_models.py` (model is now built-in)
- [x] Followed CONTRIBUTING.md guidelines for adding new models
