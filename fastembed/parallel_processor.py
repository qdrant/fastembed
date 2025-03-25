import logging
import os
from collections import defaultdict
from copy import deepcopy
from multiprocessing import get_context, Condition, Manager
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Lock
from typing import Any, Iterable, Optional, Type


max_internal_batch_size = 200


class Worker:
    @classmethod
    def start(cls, *args: Any, **kwargs: Any) -> "Worker":
        raise NotImplementedError()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        raise NotImplementedError()


def _worker(
    worker_class: Type[Worker],
    task_index: Synchronized,
    num_tasks: Synchronized,
    input_shared: list[Optional[Any]],
    output_shared: list[Optional[Any]],
    completed: Synchronized,
    lock: Lock,
    worker_id: int,
    kwargs: Optional[dict[str, Any]] = None,
) -> None:
    if kwargs is None:
        kwargs = {}

    logging.info(
        f"Reader worker: {worker_id} PID: {os.getpid()} Device: {kwargs.get('device_id', 'CPU')}"
    )
    print(
        f"Reader worker: {worker_id} PID: {os.getpid()} Device: {kwargs.get('device_id', 'CPU')}"
    )
    try:
        worker: Worker = worker_class.start(**kwargs)
        condition = Condition(lock)

        def task_iterable() -> Iterable[tuple[int, Any]]:
            while True:
                lock.acquire()
                try:
                    if completed.value >= num_tasks.value and num_tasks.value > 0:
                        print(f"Worker {worker_id} exiting: all tasks completed")
                        return

                    if task_index.value >= num_tasks.value:
                        print(
                            f"Worker {worker_id} waiting: task_index={task_index.value}, num_tasks={num_tasks.value}, completed={completed.value} <<<<<<<<<<"
                        )
                        condition.wait(timeout=0.1)
                        continue

                    my_task: int = task_index.value
                    task_index.value += 1

                    batch: Optional[Any] = input_shared[my_task % len(input_shared)]
                    print(f"Worker {worker_id} got batch {my_task}: {batch} <<<<<")
                finally:
                    lock.release()

                if batch is not None:
                    yield (my_task, batch)

        for idx, result in worker.process(task_iterable()):
            print(f"Worker {worker_id} processing complete for task {idx}")
            lock.acquire()
            try:
                output_shared[idx % len(output_shared)] = result
                completed.value += 1
                condition.notify_all()
                print(f"Worker {worker_id} completed task {idx}, completed={completed.value}")
            finally:
                lock.release()
    except Exception as e:  # pylint: disable=broad-except
        print(f"Reader worker {worker_id} failed: {e}")
        logging.exception(f"Reader worker {worker_id} failed: {e}")
    finally:
        print(f"Reader worker {worker_id} finished")
        logging.info(f"Reader worker {worker_id} finished")


class ParallelWorkerPool:
    def __init__(
        self,
        num_workers: int,
        worker: Type[Worker],
        start_method: Optional[str] = None,
        device_ids: Optional[list[int]] = None,
        cuda: bool = False,
    ):
        self.worker_class = worker
        self.num_workers = num_workers
        self.ctx: BaseContext = get_context(start_method)
        self.processes: list[BaseProcess] = []
        self.emergency_shutdown = False
        self.device_ids = device_ids
        self.cuda = cuda

        self.task_index: Optional[Synchronized[int]] = None
        self.num_tasks: Optional[Synchronized[int]] = None
        self.completed: Optional[Synchronized[int]] = None
        self.lock: Lock
        self.condition = None
        self.shared_storage_size: int = self.num_workers * max_internal_batch_size
        self.input_shared: Optional[list[Optional[Any]]] = None
        self.output_shared: Optional[list[Optional[Any]]] = None
        self.manager = Manager()

    def start(self, **kwargs: Any) -> None:
        self.task_index = self.ctx.Value("i", 0)
        self.num_tasks = self.ctx.Value("i", 0)
        self.completed = self.ctx.Value("i", 0)
        self.lock = self.ctx.Lock()
        self.condition = self.ctx.Condition(self.lock)

        self.input_shared = self.manager.list([None] * self.shared_storage_size)
        self.output_shared = self.manager.list([None] * self.shared_storage_size)

        for worker_id in range(0, self.num_workers):
            worker_kwargs = deepcopy(kwargs)
            if self.device_ids:
                device_id = self.device_ids[worker_id % len(self.device_ids)]
                worker_kwargs["device_id"] = device_id
                worker_kwargs["cuda"] = self.cuda

            assert hasattr(self.ctx, "Process")
            process = self.ctx.Process(
                target=_worker,
                args=(
                    self.worker_class,
                    self.task_index,
                    self.num_tasks,
                    self.input_shared,
                    self.output_shared,
                    self.completed,
                    self.lock,
                    worker_id,
                    worker_kwargs,
                ),
            )
            process.start()
            self.processes.append(process)

    def ordered_map(self, stream: Iterable[Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
        buffer: defaultdict[int, Any] = defaultdict(Any)  # type: ignore
        next_expected = 0

        for idx, item in self.semi_ordered_map(stream, *args, **kwargs):
            buffer[idx] = item
            while next_expected in buffer:
                yield buffer.pop(next_expected)
                next_expected += 1

    def semi_ordered_map(
        self, stream: Iterable[Any], *args: Any, **kwargs: Any
    ) -> Iterable[tuple[int, Any]]:
        try:
            self.start(**kwargs)

            total_completed: int = 0
            pushed: int = 0

            for idx, batch in enumerate(stream):
                self.check_worker_health()
                self.lock.acquire()
                try:
                    if pushed >= self.shared_storage_size:
                        while self.completed.value <= total_completed:
                            self.condition.wait(timeout=0.1)
                            self.check_worker_health()

                    print(f"pushing batch {pushed}: {batch}")
                    self.input_shared[pushed % self.shared_storage_size] = batch
                    self.num_tasks.value = pushed + 1
                    self.condition.notify_all()
                    pushed += 1
                finally:
                    self.lock.release()

                self.lock.acquire()
                try:
                    if self.completed.value > total_completed:
                        for i in range(self.num_tasks.value):
                            if self.output_shared[i] is not None:
                                print(f"yielding from scan: idx={i}")
                                yield (i, self.output_shared[i])
                                total_completed += 1
                                self.output_shared[i] = None  # clear slot (improtant)
                finally:
                    self.lock.release()

            while total_completed < pushed:
                self.check_worker_health()
                self.lock.acquire()
                try:
                    print(
                        f"drain: total_completed={total_completed}, pushed={pushed}, completed={self.completed.value} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                    )
                    if self.completed.value > total_completed:
                        for i in range(self.num_tasks.value):
                            if self.output_shared[i] is not None:
                                print(f"yielding from drain: idx={i}")
                                yield (i, self.output_shared[i])
                                total_completed += 1
                                self.output_shared[i] = None
                    elif self.completed.value < pushed:
                        self.condition.wait(timeout=0.1)
                finally:
                    self.lock.release()
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error in semi_ordered_map: {e}")
            logging.exception(f"Error in semi_ordered_map: {e}")
        finally:
            self.join()

    def check_worker_health(self) -> None:
        """
        Checks if any worker process has terminated unexpectedly
        """
        for process in self.processes:
            if not process.is_alive() and process.exitcode != 0:
                self.emergency_shutdown = True
                self.join_or_terminate()
                raise RuntimeError(
                    f"Worker PID: {process.pid} terminated unexpectedly with code {process.exitcode}"
                )

    def join_or_terminate(self, timeout: Optional[int] = 1) -> None:
        """
        Emergency shutdown
        @param timeout:
        @return:
        """
        for process in self.processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
        self.processes.clear()

    def join(self) -> None:
        for process in self.processes:
            process.join()
        self.processes.clear()

    def __del__(self) -> None:
        """
        Terminate processes if the user hasn't joined. This is necessary as
        leaving stray processes running can corrupt shared state. In brief,
        we've observed shared memory counters being reused (when the memory was
        free from the perspective of the parent process) while the stray
        workers still held a reference to them.
        For a discussion of using destructors in Python in this manner, see
        https://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python/.
        """
        for process in self.processes:
            if process.is_alive():
                process.terminate()
