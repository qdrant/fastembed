import os
from multiprocessing import get_all_start_methods
from typing import Any, Iterable
from unittest.mock import Mock

import pytest

from fastembed.common.types import Device
from fastembed.parallel_processor import ParallelWorkerPool, Worker


START_METHODS = [method for method in ("spawn", "forkserver") if method in get_all_start_methods()]


class ReportingWorker(Worker):
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    @classmethod
    def start(cls, **kwargs: Any) -> "ReportingWorker":
        return cls(kwargs)

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        for index, item in items:
            yield index, (item, self.options, os.getpid())


@pytest.mark.parametrize("start_method", START_METHODS)
@pytest.mark.parametrize("device_ids", [None, [], [2, 5]], ids=["none", "empty", "explicit"])
@pytest.mark.parametrize("cuda", [False, True, Device.CPU, Device.CUDA, Device.AUTO])
def test_workers_receive_cuda_selection_and_other_options(
    start_method: str, device_ids: list[int] | None, cuda: bool | Device
) -> None:
    pool = ParallelWorkerPool(
        1, ReportingWorker, start_method=start_method, device_ids=device_ids, cuda=cuda
    )
    other_options = {"model_name": "reporting", "config": {"batch_size": 3}}

    results = list(pool.ordered_map(range(4), **other_options))

    expected_options = {**other_options, "cuda": cuda}
    if device_ids:
        expected_options["device_id"] = device_ids[0]
    assert [item for item, _, _ in results] == list(range(4))
    for _, options, pid in results:
        assert options == expected_options
        assert options["cuda"] is cuda
        assert pid != os.getpid()


def test_device_ids_are_assigned_round_robin_at_worker_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ParallelWorkerPool(
        5, ReportingWorker, start_method="spawn", device_ids=[2, 5], cuda=Device.CUDA
    )
    process_factory = Mock()
    process_factory.return_value.is_alive.return_value = False
    monkeypatch.setattr(pool.ctx, "Process", process_factory)
    other_options = {"model_name": "reporting", "config": {"batch_size": 3}}

    try:
        pool.start(**other_options)

        worker_options = [call.kwargs["args"][-1] for call in process_factory.call_args_list]
        assert worker_options == [
            {**other_options, "cuda": Device.CUDA, "device_id": device_id}
            for device_id in [2, 5, 2, 5, 2]
        ]
    finally:
        pool.join()
        for queue in (pool.input_queue, pool.output_queue):
            if queue is not None:
                queue.close()
                queue.join_thread()


@pytest.mark.parametrize("start_method", START_METHODS)
@pytest.mark.parametrize("device_ids", [None, [], [2, 5]])
def test_pool_cuda_selection_takes_precedence_over_worker_options(
    start_method: str, device_ids: list[int] | None
) -> None:
    pool = ParallelWorkerPool(
        1, ReportingWorker, start_method=start_method, device_ids=device_ids, cuda=False
    )

    results = list(pool.ordered_map(["report"], cuda=True))

    assert len(results) == 1
    assert results[0][1]["cuda"] is False


@pytest.mark.parametrize("start_method", START_METHODS)
def test_two_workers_preserve_order_and_explicit_device_options(start_method: str) -> None:
    pool = ParallelWorkerPool(
        2, ReportingWorker, start_method=start_method, device_ids=[2, 5], cuda=False
    )

    results = list(pool.ordered_map(range(10), model_name="reporting"))

    assert [item for item, _, _ in results] == list(range(10))
    for _, options, pid in results:
        assert options == {
            "model_name": "reporting",
            "cuda": False,
            "device_id": options["device_id"],
        }
        assert options["device_id"] in {2, 5}
        assert options["cuda"] is False
        assert pid != os.getpid()
