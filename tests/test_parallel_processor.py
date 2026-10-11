import threading
from itertools import count
from multiprocessing import get_all_start_methods
from unittest.mock import Mock

import pytest

from fastembed.parallel_processor import ParallelWorkerPool, Worker


class EchoWorker(Worker):
    @classmethod
    def start(cls, **kwargs):
        return cls()

    def process(self, items):
        yield from items


@pytest.mark.parametrize("failed_queue", [1, 2])
def test_queue_creation_failure_preserves_error_and_cleans_up(monkeypatch, failed_queue):
    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn")
    queue = Mock()
    error = OSError("queue allocation failed")
    side_effect = [error] if failed_queue == 1 else [queue, error]
    monkeypatch.setattr(pool.ctx, "Queue", Mock(side_effect=side_effect))

    with pytest.raises(OSError, match="queue allocation failed") as raised:
        list(pool.ordered_map([1]))

    assert raised.value is error
    assert pool.processes == []
    if failed_queue == 2:
        queue.close.assert_called_once_with()
        queue.cancel_join_thread.assert_called_once_with()


@pytest.mark.parametrize("legacy_cleanup", [False, True])
def test_second_queue_failure_closes_real_input_queue(monkeypatch, legacy_cleanup):
    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn")
    queue = pool.ctx.Queue(pool.queue_size)
    if legacy_cleanup:
        monkeypatch.setattr(queue, "_terminate_broken", None, raising=False)
    monkeypatch.setattr(
        pool.ctx, "Queue", Mock(side_effect=[queue, OSError("output queue allocation failed")])
    )
    try:
        with pytest.raises(OSError, match="output queue allocation failed"):
            list(pool.ordered_map([1]))
        assert queue._closed
        assert queue._joincancelled
        assert queue._reader.closed
        assert queue._writer.closed
    finally:
        queue.close()
        queue._reader.close()
        queue._writer.close()


@pytest.mark.parametrize("items", [[], [1, 2, 3]])
def test_completed_mapping_stops_workers(items):
    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn")
    assert list(pool.ordered_map(items)) == items
    assert pool.processes == []


# semi_ordered_map is closed by garbage collection, so an error in its cleanup is only reported as unraisable
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_closing_partially_consumed_iterator_stops_workers():
    start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
    pool = ParallelWorkerPool(2, EchoWorker, start_method=start_method)
    # the stream never ends, so the workers never get their stop signals
    results = pool.ordered_map(count())
    assert next(results) == 0
    workers = list(pool.processes)

    # close() used to wait in join() forever, run it in a thread so a regression fails instead of
    # hanging the test session
    closer = threading.Thread(target=results.close, daemon=True)
    closer.start()
    closer.join(timeout=30)
    try:
        assert not closer.is_alive(), "closing the iterator hung"
        assert not any(worker.is_alive() for worker in workers)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
