from unittest.mock import patch

import pytest

from fastembed import SparseTextEmbedding
from fastembed.sparse.minicoil import MiniCOIL


@pytest.mark.parametrize("enable_cpu_mem_arena", [False, True])
@pytest.mark.parametrize("method", ["embed", "query_embed"])
def test_minicoil_parallel_workers_preserve_session_options(
    tmp_path, enable_cpu_mem_arena, method
):
    # Exercise public API dispatch and worker configuration without downloading a model
    # or starting inference processes. The document path is the control for queries.
    with (
        patch.object(MiniCOIL, "download_model", return_value=tmp_path),
        patch("fastembed.text.onnx_text_model.ParallelWorkerPool") as pool,
    ):
        model = SparseTextEmbedding(
            "Qdrant/minicoil-v1",
            cache_dir=str(tmp_path),
            lazy_load=True,
            enable_cpu_mem_arena=enable_cpu_mem_arena,
        )
        pool.return_value.ordered_map.return_value = iter(())

        # A list shorter than batch_size takes the serial path even with parallel set.
        list(getattr(model, method)(["first query", "second query"], batch_size=1, parallel=2))

        pool.assert_called_once()
        pool.return_value.ordered_map.assert_called_once()
        worker_options = pool.return_value.ordered_map.call_args.kwargs
        assert worker_options["enable_cpu_mem_arena"] is enable_cpu_mem_arena
        assert worker_options["is_query"] is (method == "query_embed")
