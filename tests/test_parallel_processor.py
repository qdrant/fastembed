from multiprocessing import get_context
from queue import Empty
from unittest.mock import MagicMock, call

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

    assert process.join.call_args_list == [call(timeout=1), call(timeout=1)]
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

    assert process.join.call_args_list == [call(timeout=1), call(timeout=1)]
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


def test_reused_pool_returns_to_graceful_queue_cleanup():
    pool = ParallelWorkerPool(1, EchoWorker)
    ctx = MagicMock()
    pool.ctx = ctx

    first_input_queue = MagicMock()
    first_output_queue = MagicMock()
    first_output_queue.get_nowait.return_value = (0, "first result")
    second_input_queue = MagicMock()
    second_output_queue = MagicMock()
    second_output_queue.get_nowait.return_value = (0, "second result")
    first_process = MagicMock()
    first_process.is_alive.return_value = True
    second_process = MagicMock()

    ctx.Queue.side_effect = [
        first_input_queue,
        first_output_queue,
        second_input_queue,
        second_output_queue,
    ]
    ctx.Value.side_effect = [
        get_context().Value("i", 1),
        get_context().Value("i", 1),
    ]
    ctx.Process.side_effect = [first_process, second_process]

    first_results = pool.ordered_map(["first input"])
    assert next(first_results) == "first result"
    first_results.close()

    assert pool.emergency_shutdown is True
    first_input_queue.cancel_join_thread.assert_called_once_with()
    first_output_queue.cancel_join_thread.assert_called_once_with()

    assert list(pool.semi_ordered_map(["second input"])) == [(0, "second result")]

    assert pool.emergency_shutdown is False
    second_input_queue.join_thread.assert_called_once_with()
    second_output_queue.join_thread.assert_called_once_with()
    second_input_queue.cancel_join_thread.assert_not_called()
    second_output_queue.cancel_join_thread.assert_not_called()
