import pytest

from fastembed import TextEmbedding

CACHE_DIR = "../model_cache"


@pytest.mark.skip(reason="Requires a multi-gpu server to be launched on CI")
@pytest.mark.parametrize("device_id", [0, 1])
def test_gpu_via_providers(device_id):
    embedding_model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        providers=[("CUDAExecutionProvider", {"device_id": device_id})],
        cache_dir=CACHE_DIR,
    )

    docs = ["hello world", "flag embedding"]

    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_id)


@pytest.mark.skip(reason="Requires a multi-gpu server to be launched on CI")
@pytest.mark.parametrize("device_ids", [[0], [1], [0, 1, 2, 3]])
def test_gpu_cuda_device_ids(device_ids):
    embedding_model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )

    docs = ["hello world", "flag embedding"]

    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_ids[0])


@pytest.mark.skip(reason="Requires a multi-gpu server to be launched on CI")
@pytest.mark.parametrize(
    "device_ids,parallel", [([1], None), ([1], 1), ([1], 2), ([0, 1, 2, 3], 4)]
)
def test_multi_gpu_parallel_inference(device_ids, parallel):
    embedding_model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
        lazy_load=True,
    )
    docs = ["hello world", "flag embedding"] * 100
    batch_size = 5

    list(embedding_model.embed(docs, batch_size=batch_size, parallel=len(device_ids)))
