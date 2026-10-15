import threading
from itertools import count
from multiprocessing import get_all_start_methods

import pytest

from fastembed.parallel_processor import ParallelWorkerPool, Worker


class EchoWorker(Worker):
    @classmethod
    def start(cls, **kwargs):
        return cls()

    def process(self, items):
        yield from items


@pytest.mark.parametrize("num_workers", [0, -1, -2])
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_non_positive_worker_counts_are_rejected(num_workers):
    with pytest.raises(ValueError, match="num_workers must be greater than zero"):
        ParallelWorkerPool(num_workers, EchoWorker)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_positive_worker_counts_preserve_ordered_results(num_workers):
    start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
    pool = ParallelWorkerPool(num_workers, EchoWorker, start_method=start_method)

    assert list(pool.ordered_map([3, 1, 2])) == [3, 1, 2]
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
