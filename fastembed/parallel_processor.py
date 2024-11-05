import logging
import os
from collections import defaultdict
from enum import Enum
from multiprocessing import Queue, get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized as BaseValue
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from copy import deepcopy

# Single item should be processed in less than:
processing_timeout = 10 * 60  # seconds

max_internal_batch_size = 200


class QueueSignals(str, Enum):
    stop = "stop"
    confirm = "confirm"
    error = "error"


class Worker:
    @classmethod
    def start(cls, *args: Any, **kwargs: Any) -> "Worker":
        raise NotImplementedError()

    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        raise NotImplementedError()


def _worker(
    worker_class: Type[Worker],
    input_queue: Queue,
    output_queue: Queue,
    num_active_workers: BaseValue,
    worker_id: int,
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    A worker that pulls data pints off the input queue, and places the execution result on the output queue.
    When there are no data pints left on the input queue, it decrements
    num_active_workers to signal completion.
    """

    if kwargs is None:
        kwargs = {}

    logging.info(
        f"Reader worker: {worker_id} PID: {os.getpid()} Device: {kwargs.get('device_id', 'CPU')}"
    )
    try:
        worker = worker_class.start(**kwargs)

        # Keep going until you get an item that's None.
        def input_queue_iterable() -> Iterable[Any]:
            while True:
                item = input_queue.get()
                if item == QueueSignals.stop:
                    break
                yield item

        for processed_item in worker.process(input_queue_iterable()):
            output_queue.put(processed_item)
    except Exception as e:  # pylint: disable=broad-except
        logging.exception(e)
        output_queue.put(QueueSignals.error)
    finally:
        # It's important that we close and join the queue here before
        # decrementing num_active_workers. Otherwise our parent may join us
        # before the queue's feeder thread has passed all buffered items to
        # the underlying pipe resulting in a deadlock.
        #
        # See:
        # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#pipes-and-queues
        # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#programming-guidelines
        input_queue.close()
        output_queue.close()
        input_queue.join_thread()
        output_queue.join_thread()

        with num_active_workers.get_lock():
            num_active_workers.value -= 1

        logging.info(f"Reader worker {worker_id} finished")


class ParallelWorkerPool:
    def __init__(
        self,
        num_workers: int,
        worker: Type[Worker],
        start_method: Optional[str] = None,
        device_ids: Optional[List[int]] = None,
        cuda: bool = False,
    ):
        self.worker_class = worker
        self.num_workers = num_workers
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self.ctx: BaseContext = get_context(start_method)
        self.processes: List[BaseProcess] = []
        self.queue_size = self.num_workers * max_internal_batch_size
        self.emergency_shutdown = False
        self.device_ids = device_ids
        self.cuda = cuda
        self.num_active_workers: Optional[BaseValue] = None

    def start(self, **kwargs: Any) -> None:
        self.input_queue = self.ctx.Queue(self.queue_size)
        self.output_queue = self.ctx.Queue(self.queue_size)

        ctx_value = self.ctx.Value("i", self.num_workers)
        assert isinstance(ctx_value, BaseValue)
        self.num_active_workers = ctx_value

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
                    self.input_queue,
                    self.output_queue,
                    self.num_active_workers,
                    worker_id,
                    worker_kwargs,
                ),
            )
            process.start()
            self.processes.append(process)

    def ordered_map(self, stream: Iterable[Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
        buffer = defaultdict(Any)
        next_expected = 0

        for idx, item in self.semi_ordered_map(stream, *args, **kwargs):
            buffer[idx] = item
            while next_expected in buffer:
                yield buffer.pop(next_expected)
                next_expected += 1

    def semi_ordered_map(
        self, stream: Iterable[Any], *args: Any, **kwargs: Any
    ) -> Iterable[Tuple[int, Any]]:
        try:
            self.start(**kwargs)

            assert self.input_queue is not None, "Input queue was not initialized"
            assert self.output_queue is not None, "Output queue was not initialized"

            pushed = 0
            read = 0
            for idx, item in enumerate(stream):
                self.check_worker_health()
                if pushed - read < self.queue_size:
                    try:
                        out_item = self.output_queue.get_nowait()
                    except Empty:
                        out_item = None
                else:
                    try:
                        out_item = self.output_queue.get(timeout=processing_timeout)
                    except Empty as e:
                        self.join_or_terminate()
                        raise e

                if out_item is not None:
                    if out_item == QueueSignals.error:
                        self.join_or_terminate()
                        raise RuntimeError("Thread unexpectedly terminated")
                    yield out_item
                    read += 1

                self.input_queue.put((idx, item))
                pushed += 1

            for _ in range(self.num_workers):
                self.input_queue.put(QueueSignals.stop)

            while read < pushed:
                self.check_worker_health()
                out_item = self.output_queue.get(timeout=processing_timeout)
                if out_item == QueueSignals.error:
                    self.join_or_terminate()
                    raise RuntimeError("Thread unexpectedly terminated")
                yield out_item
                read += 1
        finally:
            assert self.input_queue is not None, "Input queue is None"
            assert self.output_queue is not None, "Output queue is None"
            self.join()
            self.input_queue.close()
            self.output_queue.close()
            if self.emergency_shutdown:
                self.input_queue.cancel_join_thread()
                self.output_queue.cancel_join_thread()
            else:
                self.input_queue.join_thread()
                self.output_queue.join_thread()

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
