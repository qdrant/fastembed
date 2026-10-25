"""Focused, offline tests for the opt-in bounded CPU inference path.

Run from the FastEmbed repository root after applying the patch:
  python -m pytest -q tests/test_cpu_concurrent_batches.py
"""

import threading
import time

import numpy as np
import pytest

from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.text.onnx_text_model import OnnxTextModel


class FakeSession:
    def __init__(self, providers=None):
        self.providers = providers or ["CPUExecutionProvider"]

    def get_providers(self):
        return self.providers


class DummyModel(OnnxTextModel[int]):
    """Deliberately returns batches in a different completion order."""

    def __init__(self, *, providers=None, fail_on=None):
        super().__init__()
        self.model = FakeSession(providers)
        self.tokenizer = object()  # avoid loading any real model files
        self.fail_on = fail_on
        self.calls = []
        self.lock = threading.Lock()

    def _get_worker_class(self):
        raise NotImplementedError

    def onnx_embed(self, documents, **kwargs):
        numbers = [int(d) for d in documents]
        with self.lock:
            self.calls.append(numbers[0])
        # Force small, deterministic skew. All work is bounded.
        time.sleep(0.001 * (numbers[0] % 3))
        if self.fail_on in numbers:
            raise RuntimeError("inference failure")
        return OnnxOutputContext(
            model_output=np.asarray(numbers, dtype=np.int64),
            attention_mask=None,
            input_ids=None,
        )

    def _post_process_onnx_output(self, output, **kwargs):
        return iter(output.model_output.tolist())


def encode(model, docs, inference_workers=1, parallel=None, batch_size=2):
    return list(
        model._embed_documents(
            model_name="dummy",
            cache_dir=".",
            documents=docs,
            batch_size=batch_size,
            parallel=parallel,
            inference_workers=inference_workers,
        )
    )


def test_order_and_values_match_existing_serial_path():
    docs = [str(i) for i in range(20)]
    serial = encode(DummyModel(), docs, inference_workers=1)
    concurrent = encode(DummyModel(), docs, inference_workers=4)
    assert serial == concurrent == list(range(20))


def test_generator_is_bounded_and_lazy():
    state = {"consumed": 0}

    def unlimited():
        while True:
            current = state["consumed"]
            state["consumed"] += 1
            yield str(current)

    output = DummyModel()._embed_documents(
        model_name="dummy",
        cache_dir=".",
        documents=unlimited(),
        batch_size=2,
        parallel=None,
        inference_workers=3,
    )
    assert next(output) == 0
    assert state["consumed"] <= 12  # 2 * 3 outstanding batches of size 2
    output.close()  # Must not hang or consume unbounded input


def test_empty_input_returns_empty_result():
    assert encode(DummyModel(), [], inference_workers=3) == []


@pytest.mark.parametrize("bad", [0, -1, False, True, 1.5, "4"])
def test_invalid_worker_counts_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        encode(DummyModel(), ["1", "2", "3"], inference_workers=bad)


def test_existing_parallel_processes_not_combined_with_threads():
    with pytest.raises(ValueError, match="cannot be combined"):
        encode(DummyModel(), ["1", "2", "3"], inference_workers=2, parallel=2)


def test_non_cpu_provider_is_rejected():
    with pytest.raises(ValueError, match="CPU-only"):
        encode(
            DummyModel(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]),
            ["1", "2", "3"],
            inference_workers=2,
        )


def test_exception_from_inference_is_propagated():
    with pytest.raises(RuntimeError, match="inference failure"):
        encode(DummyModel(fail_on=5), [str(i) for i in range(10)], inference_workers=3)


def test_non_cpu_provider_is_rejected_even_for_short_input():
    # The CPU-only guarantee must not silently disappear for short lists.
    with pytest.raises(ValueError, match="CPU-only"):
        encode(
            DummyModel(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]),
            ["1"],
            inference_workers=3,
            batch_size=16,
        )


def test_early_close_cancels_queued_batches(monkeypatch):
    import fastembed.text.onnx_text_model as target

    created = []

    class RecordedFuture:
        def __init__(self, result):
            self.value = result
            self.cancel_attempted = False

        def result(self):
            return self.value

        def cancel(self):
            self.cancel_attempted = True
            return True

    class RecordedExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.submissions = []
            self.shutdown_kwargs = None
            created.append(self)

        def submit(self, fn, *args, **kwargs):
            future = RecordedFuture(fn(*args, **kwargs))
            self.submissions.append(future)
            return future

        def shutdown(self, **kwargs):
            self.shutdown_kwargs = kwargs

    monkeypatch.setattr(target, "ThreadPoolExecutor", RecordedExecutor)
    model = DummyModel()
    output = model._embed_documents(
        model_name="dummy",
        cache_dir=".",
        documents=(str(i) for i in range(100)),
        batch_size=2,
        inference_workers=2,
    )
    assert next(output) == 0
    output.close()
    assert len(created) == 1
    executor = created[0]
    assert executor.shutdown_kwargs == {"wait": True, "cancel_futures": True}
    assert any(f.cancel_attempted for f in executor.submissions)


def test_real_cpu_onnx_concurrent_embeddings_match_serial():
    from fastembed import TextEmbedding

    docs = [f"Example document {i} about embedding inference." for i in range(64)]

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        providers=["CPUExecutionProvider"],
        threads=1,
    )

    expected = np.stack(list(model.embed(docs, batch_size=8)))
    actual = np.stack(list(model.embed(docs, batch_size=8, inference_workers=4)))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
