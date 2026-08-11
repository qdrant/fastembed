from queue import Empty
from unittest.mock import MagicMock

import pytest

from fastembed.parallel_processor import ParallelWorkerPool, Worker


class EchoWorker(Worker):
    @classmethod
    def start(cls, **kwargs):
        return cls()

    def process(self, items):
        yield from items


def make_stubbed_pool(monkeypatch):
    pool = ParallelWorkerPool(1, EchoWorker)
    input_queue = MagicMock()
    output_queue = MagicMock()
    output_queue.get_nowait.side_effect = Empty
    output_queue.get.return_value = (0, "result")
    process = MagicMock()
    process.is_alive.return_value = True

    def start(**kwargs):
        pool.input_queue = input_queue
        pool.output_queue = output_queue
        pool.processes = [process]

    monkeypatch.setattr(pool, "start", start)
    return pool, input_queue, output_queue, process


def test_closing_partial_parallel_iterator_terminates_workers(monkeypatch):
    pool, input_queue, output_queue, process = make_stubbed_pool(monkeypatch)
    results = pool.ordered_map(["input"])

    assert next(results) == "result"
    results.close()

    process.join.assert_called_once_with(timeout=1)
    process.terminate.assert_called_once_with()
    input_queue.cancel_join_thread.assert_called_once_with()
    output_queue.cancel_join_thread.assert_called_once_with()


def test_parallel_iterator_terminates_workers_when_input_raises(monkeypatch):
    pool, input_queue, output_queue, process = make_stubbed_pool(monkeypatch)

    def failing_stream():
        yield "input"
        raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError, match="stream failed"):
        list(pool.semi_ordered_map(failing_stream()))

    process.join.assert_called_once_with(timeout=1)
    process.terminate.assert_called_once_with()
    input_queue.cancel_join_thread.assert_called_once_with()
    output_queue.cancel_join_thread.assert_called_once_with()


def test_exhausted_parallel_iterator_joins_workers_gracefully(monkeypatch):
    pool, input_queue, output_queue, process = make_stubbed_pool(monkeypatch)

    assert list(pool.semi_ordered_map(["input"])) == [(0, "result")]

    process.join.assert_called_once_with()
    process.terminate.assert_not_called()
    input_queue.join_thread.assert_called_once_with()
    output_queue.join_thread.assert_called_once_with()
