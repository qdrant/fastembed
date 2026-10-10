# FastEmbed engineering report

Date: 10 October 2026 (Asia/Calcutta).

## Delivered change

Selected [qdrant/fastembed](https://github.com/qdrant/fastembed), created the
GitHub fork [irulappan151204/fastembed](https://github.com/irulappan151204/fastembed),
and implemented focused model-download reliability and diagnostic-security fixes.
The work is on `fix/model-download-resources`. The implementation and nine tests
are in commit `cab30737fc8234f903f09898d53e9ef094e9211c`; the README and this
report are committed separately. No changes were pushed to upstream.

## Selection and repository evaluation

All seven repositories exposed by the connected account were screened using
metadata and repository trees, with deeper inspection of the AI candidate and
the Python reporting applications' dependencies.

| Repository | Assessment |
| --- | --- |
| `Chatting-with-Local-RAG` | The clearest personal AI project: MIT-licensed PDF RAG using LangChain, Chroma, and Ollama. Last push December 2024; a single notebook, no automated tests, invalid package names such as `OllamaEmbeddings` in requirements, and deprecated imports. Reviving it would require establishing a maintained application structure and dependency baseline. |
| `DSR_2.0` and a private reporting repository | Flask/SQL reporting applications. Their dependency lists and inspected structure show reporting rather than an established AI pipeline. |
| `rafithub` | TypeScript fitness website. |
| `newPortfolio` and a private portfolio repository | TypeScript portfolio websites. |
| `irulappan151204` | GitHub profile configuration. |

Three external Python projects were compared: FastEmbed, NeuML's txtai, and
AnswerDotAI's rerankers. FastEmbed was pushed on 9 October 2026, has about
3,240 stars in the inspected snapshot, and combines an active maintainer,
compact codebase, real RAG use, and testable infrastructure. txtai was also
active but has a substantially broader orchestration and model surface;
rerankers' last observed push was December 2025. FastEmbed offered the best
balance for a bounded, verifiable contribution. This is an engineering judgment
about these inspected candidates, not a claim to have ranked all AI projects.

Upstream issue review included the
[tuple reranking bug](https://github.com/qdrant/fastembed/issues/796),
[non-finite embedding report](https://github.com/qdrant/fastembed/issues/688),
[CPU/GPU runtime collision](https://github.com/qdrant/fastembed/issues/608),
and [Hub 2.x support](https://github.com/qdrant/fastembed/issues/742).
The tuple and NaN bugs already have open pull requests
[797](https://github.com/qdrant/fastembed/pull/797) and
[691](https://github.com/qdrant/fastembed/pull/691), so their fixes were not
duplicated. The delivered bugs were found by inspecting the current downloader.

## Architecture, quality, dependencies, and documentation

The baseline is FastEmbed 0.9.0, commit
`d076f083519746d132888079e6e790f2193c9c08`.

Public facades select registered dense, sparse, image, late-interaction, and
cross-encoder implementations. Shared model management resolves cached files,
downloads weights/configuration, verifies metadata, and stages archives.
Tokenizers and image processors construct ONNX inputs; ONNX Runtime performs
inference. A worker pool supports ordered multiprocessing. Common numeric
helpers and MUVERA handle postprocessing.

The codebase uses typed interfaces, dataclass model descriptions, pytest,
Ruff, mypy, and CI across Python 3.10–3.14. Existing cache validation, archive
path checks, bounded HTTP timeouts, and staging logic are valuable safeguards.
The main test suite depends heavily on downloaded models and some hardware.
The downloader lacked exception-safe HTTP response ownership and exposed
custom source URLs in three diagnostics. The README's usage examples were
useful, but this fork adds a reproducible contributor workflow and configuration
guidance. No architecture rewrite or inference-algorithm changes were needed.

Runtime and test dependencies were installed from the existing `poetry.lock`.
Observed versions include NumPy 2.4.6, Requests 2.34.2, ONNX Runtime 1.30.0,
Hugging Face Hub 1.33.0, Tokenizers 0.23.2, Pillow 12.3.0, pytest 9.1.1, and
mypy 2.4.0. The upstream pre-commit Ruff pin, 0.3.4, was used for final lint
and formatting checks. `pyproject.toml` and `poetry.lock` remain unchanged.

## Reproduced issues and fixes

1. **Unclosed HTTP streams on failure.** Real local HTTP responses remained open
   after 403/404 responses, malformed content-length values, and destination-open
   errors. The response now has a context manager covering status validation,
   header parsing, progress reporting, and file writing. Existing exception types,
   timeouts, chunk sizes, return values, and successful file bytes are preserved.
2. **Sensitive source URLs in diagnostics.** Missing/zero-length warnings,
   fallback-download logs, and missing-model-directory errors exposed a synthetic
   signed-URL marker. Those diagnostics now omit the source URL; errors identify
   the model and the warning uses the existing Loguru logger.
3. **Development-tool advisories.** A temporary Poetry 2.2.1 environment brought
   in Dulwich 0.24.10. pip-audit reported 18 advisory entries: 14 for Dulwich and
   four for Poetry, including overlapping advisory aliases. These were tooling
   findings, not newly identified FastEmbed runtime vulnerabilities. Upgrading
   the isolated tooling to Poetry 2.5.1 brought Dulwich 1.2.17 and resolved the
   audit findings. The README pins the verified Poetry version. Poetry's
   [published wheel-path advisory](https://github.com/python-poetry/poetry/security/advisories/GHSA-2599-h6xx-hpxp)
   independently confirms the risk in the older tooling.

Nine HTTP regression/compatibility cases exercise real local transport rather
than mocked responses. They cover the failure paths, successful bytes and
closure, absent/zero length, fallback logs, and incomplete archives. Captured
responses and logger sinks are cleaned up by test fixtures. The local-download
test overrides external-model offline mode for isolation.

## Verification evidence

Windows, CPython 3.11.15, CPU inference. No API keys or paid inference services
were used.

| Verification | Observed result |
| --- | --- |
| Initial seven download tests, before production changes | 6 failed, 1 passed; four open-stream assertions and two URL-disclosure assertions reproduced the bugs. |
| Two added diagnostics tests, before their fixes | Both reproduced URL disclosure after fixing a logger-capture issue in the fixture. |
| Focused tests: download, model management, common utilities, image transforms, worker pool | 32 passed; the final run after the tooling upgrade took 8.37 seconds. |
| Existing `tests/test_text_cross_encoder.py`, `CI=1`, online model download | 6 passed in 52.72 seconds with the real `Xenova/ms-marco-MiniLM-L-6-v2` model: scores, batching, lazy loading, parallel inference, token counting, and session configuration. |
| Full suite, `HF_HUB_OFFLINE=1`, `CI=1`, dedicated cache | 57 passed, 23 skipped, 74 failed, 39 setup errors, 4 warnings. All 113 remaining failure/error entries report an uncached model. Full online-suite success is not claimed. |
| Ruff 0.3.4, explicit repository configuration | Check and format check passed for both changed Python files. |
| mypy with upstream CI flags | Success across 66 source files. |
| `uv pip check` | All 76 installed packages compatible. |
| pip-audit after tooling upgrade | 76 packages audited, zero known advisory entries, no skipped packages. This is a database snapshot, not proof of complete security. |
| Git checks | Diff whitespace check passed; license, notice, dependency declarations, and lockfile unchanged. |

The full offline run also exposed one environment-sensitive assertion in a new
test. Its fixture was corrected, and the full run was repeated; the final counts
above include the correction. The four warnings concern the upstream
ColModernVBERT model update. The accompanying offline log names every failed
and errored test; initial/final dependency-audit JSON files preserve that audit's
progression.

## Reproduction and local environment

Use the installation, configuration, and commands in the updated README.
Poetry 2.5.1 successfully installed the unchanged lockfile. This host initially
omitted `PROCESSOR_ARCHITECTURE`, causing Poetry's marker parser to fail;
setting the process variable to its actual value, `AMD64`, resolved that issue.

Git could not create files under Documents directly. The cloned source is in
the chat workspace's `work/fastembed`; its `.git` pointer refers to metadata in
`%TEMP%/codex-fastembed-20261010/.git`. The isolated virtual environment is
`%TEMP%/codex-fastembed-venv`. Caches were disabled or placed in temporary
directories. A fresh clone in a writable development directory is recommended
for continued work; the delivered Git bundle is a portable recovery copy.

## Licensing and security boundaries

The original Apache-2.0 `LICENSE`, Qdrant attribution, and third-party `NOTICE`
were retained without changes. A modified-file notice was added to the changed
source, and the README identifies the upstream revision and modifications.
Model weights have separate licenses, including restricted licenses mentioned
in `NOTICE`; the library's license does not replace them.

GitHub's connector lacks a fork operation and the browser was signed out, so
the fork was created through GitHub's API using the existing Git Credential
Manager credential in memory. It was not printed, embedded in a URL, or saved
to a file. Git remotes use ordinary HTTPS URLs. Production applications and
databases were not accessed or modified.

Kluster startup reported `connectionError`; automatic-review and
dependency-check tools were unavailable. No Kluster review result is claimed.
Testing, static checks, and the dependency-advisory audit are recorded separately.

## Remaining limitations and next steps

- Run the full online suite with all required licensed model assets and GPU
  hardware, and the supported Python/OS matrix. The six real inference tests
  cover one representative reranker, not every model family.
- The delivered diagnostics fixes do not sanitize all third-party HTTP
  exceptions. Avoid indiscriminately logging raw exceptions or configuration;
  a wider diagnostic audit would be a separate change.
- Monitor upstream PRs 691/797 and the CPU/GPU dependency and Hub-version
  issues rather than applying broad compatibility changes without their tests.
- Keep the development tooling pin and advisory checks current. No throughput
  improvement or memory-performance benchmark is claimed.
- Review this fork's branch before proposing an upstream contribution. No
  upstream pull request or production deployment was made.
