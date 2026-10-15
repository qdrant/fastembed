import os
import time
from multiprocessing import get_all_start_methods
from pathlib import Path
from queue import Empty

import pytest

import fastembed.parallel_processor as processor
from fastembed.parallel_processor import ParallelWorkerPool, Worker


class ControlledWorker(Worker):
    @classmethod
    def start(cls, marker, crash=False, delay=0, **kwargs):
        worker = cls()
        worker.marker = marker
        worker.crash = crash
        worker.delay = delay
        return worker

    def process(self, items):
        for item in items:
            deadline = time.monotonic() + 30
            while not Path(self.marker).exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Parent did not reach its result wait")
                time.sleep(0.01)
            if self.crash:
                os._exit(23)
            time.sleep(self.delay)
            yield item


@pytest.fixture
def make_pool(monkeypatch, tmp_path):
    workers = []
    monkeypatch.setattr(processor, "max_internal_batch_size", 1)

    def create(start_method="spawn"):
        marker = tmp_path / "waiting"
        pool = ParallelWorkerPool(1, ControlledWorker, start_method=start_method)
        original_start = pool.start

        def start(**kwargs):
            original_start(marker=str(marker), **kwargs)
            workers.extend(pool.processes)
            queue = pool.output_queue
            assert queue is not None
            original_get = queue.get

            def get(block=True, timeout=None):
                if block and timeout is not None:
                    marker.touch()
                return original_get(block=block, timeout=timeout)

            monkeypatch.setattr(queue, "get", get)

        monkeypatch.setattr(pool, "start", start)
        return pool

    yield create
    for worker in workers:
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=5)


@pytest.mark.parametrize(
    "start_method",
    [method for method in ("spawn", "forkserver") if method in get_all_start_methods()],
)
@pytest.mark.parametrize("backpressure", [False, True])
def test_worker_crash_during_result_wait(make_pool, monkeypatch, start_method, backpressure):
    monkeypatch.setattr(processor, "processing_timeout", 10)
    pool = make_pool(start_method)
    # A second item fills the input window, exercising the backpressure read rather than the drain.
    items = range(2 if backpressure else 1)
    with pytest.raises(RuntimeError, match="terminated unexpectedly with code 23"):
        list(pool.ordered_map(items, crash=True))
    assert pool.processes == []


def test_healthy_worker_can_finish_after_a_poll(make_pool, monkeypatch):
    monkeypatch.setattr(processor, "processing_timeout", 10)
    pool = make_pool()
    assert list(pool.ordered_map([7], delay=1.2)) == [7]
    assert pool.processes == []


def test_healthy_worker_keeps_the_overall_timeout(make_pool, monkeypatch):
    monkeypatch.setattr(processor, "processing_timeout", 1.2)
    pool = make_pool()
    with pytest.raises(Empty):
        list(pool.ordered_map([7], delay=5))
    assert pool.processes == []
