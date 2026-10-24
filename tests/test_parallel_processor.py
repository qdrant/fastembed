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


@pytest.mark.parametrize("failure_stage", ["value", "first_worker", "second_worker"])
def test_startup_failure_closes_real_queue_endpoints(monkeypatch, failure_stage):
    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn")
    error = OSError("pool startup failed")
    workers = []
    if failure_stage == "value":
        monkeypatch.setattr(pool.ctx, "Value", Mock(side_effect=error))
    else:
        create_process = pool.ctx.Process

        def process_factory(*args, **kwargs):
            process = create_process(*args, **kwargs)
            workers.append(process)
            failed_worker = 1 if failure_stage == "first_worker" else 2
            if len(workers) == failed_worker:
                monkeypatch.setattr(process, "start", Mock(side_effect=error))
            return process

        monkeypatch.setattr(pool.ctx, "Process", process_factory)

    try:
        with pytest.raises(OSError, match="pool startup failed") as raised:
            list(pool.ordered_map([1]))
        assert raised.value is error
        assert pool.processes == []
        assert not any(worker.is_alive() for worker in workers)
        for queue in (pool.input_queue, pool.output_queue):
            assert queue is not None
            assert queue._closed
            assert queue._joincancelled
            assert queue._reader.closed
            assert queue._writer.closed
    finally:
        for queue in (pool.input_queue, pool.output_queue):
            if queue is not None:
                queue.close()
                queue._reader.close()
                queue._writer.close()


@pytest.mark.parametrize("items", [[], [1, 2, 3]])
def test_completed_mapping_stops_workers(items):
    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn")
    assert list(pool.ordered_map(items)) == items
    assert pool.processes == []
    for queue in (pool.input_queue, pool.output_queue):
        assert queue is not None
        assert queue._reader.closed
        assert queue._writer.closed


# semi_ordered_map is closed by garbage collection, so an error in its cleanup is only reported as unraisable
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize(
    "start_method",
    [method for method in ("spawn", "forkserver") if method in get_all_start_methods()],
)
def test_closing_partially_consumed_iterator_stops_workers(start_method):
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
        assert pool.output_queue is not None
        assert pool.output_queue._reader.closed
        assert pool.output_queue._writer.closed
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
