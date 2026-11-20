"""Keep a slow early batch from buffering the rest of a streaming corpus."""

import threading
import time
from collections import deque
from multiprocessing import get_all_start_methods
from pathlib import Path
from queue import Empty

import pytest

from fastembed.parallel_processor import ParallelWorkerPool, QueueSignals, Worker


class EchoWorker(Worker):
    @classmethod
    def start(cls, **kwargs):
        return cls()

    def process(self, items):
        yield from items


class DelayedQueue:
    """Deterministic schedule: batch zero finishes only when the caller waits for output."""

    def __init__(self):
        self.pending = deque()

    def put(self, item, **kwargs):
        if item != QueueSignals.stop:
            self.pending.append(item)

    def get_nowait(self):
        for item in self.pending:
            if item[0] != 0:
                self.pending.remove(item)
                return item
        raise Empty

    def get(self, **kwargs):
        if not self.pending:
            raise Empty
        return self.pending.popleft()

    def close(self):
        pass

    def join_thread(self):
        pass

    def cancel_join_thread(self):
        pass

    def _terminate_broken(self):
        pass


@pytest.mark.parametrize("size", [0, 1, 4, 5, 23])
def test_ordered_map_bounds_submitted_work(monkeypatch, size):
    pool = ParallelWorkerPool(2, EchoWorker)
    pool.queue_size = 4
    queue = DelayedQueue()

    def start(**kwargs):
        pool.input_queue = queue
        pool.output_queue = queue

    monkeypatch.setattr(pool, "start", start)
    delivered = []

    def source():
        for index in range(size):
            # The for loop may fetch one item before waiting for submission capacity.
            assert index + 1 - len(delivered) <= pool.queue_size + 1
            yield index

    for value in pool.ordered_map(source()):
        delivered.append(value)  # noqa: PERF402 - source observes each delivered item
    assert delivered == list(range(size))


def test_unordered_map_can_deliver_before_batch_zero(monkeypatch):
    pool = ParallelWorkerPool(2, EchoWorker)
    pool.queue_size = 4
    queue = DelayedQueue()

    def start(**kwargs):
        pool.input_queue = queue
        pool.output_queue = queue

    monkeypatch.setattr(pool, "start", start)
    results = list(pool.semi_ordered_map(range(10)))
    assert results[0] == (1, 1)
    assert sorted(results) == list(enumerate(range(10)))


class DelayedFirstWorker(Worker):
    @classmethod
    def start(cls, directory, **kwargs):
        worker = cls()
        worker.directory = Path(directory)
        return worker

    def process(self, items):
        for index, value in items:
            if index == 0:
                deadline = time.monotonic() + 15
                while not (self.directory / "release").exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("test did not release batch zero")
                    time.sleep(0.01)
            elif index == 3:
                (self.directory / "ready").touch()
            yield index, value


@pytest.mark.parametrize(
    "start_method",
    [method for method in ("spawn", "forkserver") if method in get_all_start_methods()],
)
def test_real_workers_bound_source_lead_and_preserve_order(tmp_path, start_method):
    pool = ParallelWorkerPool(2, DelayedFirstWorker, start_method=start_method)
    pool.queue_size = 4
    delivered = []
    leads = []
    errors = []

    def source():
        for index in range(60):
            leads.append(index + 1 - len(delivered))
            yield index

    def consume():
        try:
            for value in pool.ordered_map(source(), directory=str(tmp_path)):
                delivered.append(value)  # noqa: PERF402 - source observes each delivered item
        except (AssertionError, RuntimeError, OSError) as error:
            errors.append(error)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    workers = []
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "ready").exists() and consumer.is_alive():
            assert time.monotonic() < deadline, "workers did not reach the delayed batch"
            time.sleep(0.01)
        workers = list(pool.processes)
        # Leave the first worker gated while the second can drain all later work.
        time.sleep(0.2)
    finally:
        workers = list(pool.processes)
        (tmp_path / "release").touch()
        consumer.join(timeout=15)
        workers.extend(worker for worker in pool.processes if worker not in workers)
        leaked_workers = [worker for worker in workers if worker.is_alive()]
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
    assert not consumer.is_alive()
    assert not errors
    assert not leaked_workers
    assert delivered == list(range(60))
    assert max(leads) <= pool.queue_size + 1
    assert not pool.processes


@pytest.mark.parametrize("ordered", [False, True])
def test_input_failure_stops_workers(ordered):
    pool = ParallelWorkerPool(2, EchoWorker, start_method="spawn")
    workers = []

    def source():
        workers.extend(pool.processes)
        yield 1
        raise ValueError("input failed")

    mapper = pool.ordered_map if ordered else pool.semi_ordered_map
    with pytest.raises(ValueError, match="input failed"):
        list(mapper(source()))
    assert not pool.processes
    assert not any(worker.is_alive() for worker in workers)


class FailingWorker(EchoWorker):
    def process(self, items):
        for index, value in items:
            if index == 2:
                raise ValueError("worker failed")
            yield index, value


@pytest.mark.parametrize("ordered", [False, True])
def test_worker_failure_stops_pool(ordered):
    pool = ParallelWorkerPool(2, FailingWorker, start_method="spawn")
    mapper = pool.ordered_map if ordered else pool.semi_ordered_map
    with pytest.raises(RuntimeError, match="Thread unexpectedly terminated"):
        list(mapper(range(10)))
    assert not pool.processes


class KeywordWorker(EchoWorker):
    @classmethod
    def start(cls, ordered, **kwargs):
        assert ordered == "worker-option"
        return cls()


def test_worker_keyword_is_not_consumed_by_internal_ordering():
    pool = ParallelWorkerPool(1, KeywordWorker, start_method="spawn")
    assert list(pool.ordered_map(range(3), ordered="worker-option")) == [0, 1, 2]
