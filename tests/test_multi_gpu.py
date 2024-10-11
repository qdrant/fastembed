import pytest
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
    ImageEmbedding,
)
from tests.config import TEST_MISC_DIR

CACHE_DIR = "../model_cache"


@pytest.mark.skip(reason="Requires a multi-gpu server")
@pytest.mark.parametrize("device_id", [None, 0, 1])
def test_gpu_via_providers(device_id):
    docs = ["hello world", "flag embedding"]

    device_id = device_id if device_id is not None else 0
    providers = (
        ["CUDAExecutionProvider"]
        if device_id is None
        else [("CUDAExecutionProvider", {"device_id": device_id})]
    )
    embedding_model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        providers=providers,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_id)

    embedding_model = SparseTextEmbedding(
        "prithvida/Splade_PP_en_v1",
        providers=providers,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_id)

    embedding_model = SparseTextEmbedding(
        "Qdrant/bm42-all-minilm-l6-v2-attentions",
        providers=providers,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_id)

    embedding_model = LateInteractionTextEmbedding(
        "colbert-ir/colbertv2.0",
        providers=providers,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_id)

    embedding_model = ImageEmbedding(
        model_name="Qdrant/clip-ViT-B-32-vision",
        providers=providers,
        cache_dir=CACHE_DIR,
    )
    images = [
        TEST_MISC_DIR / "image.jpeg",
        str(TEST_MISC_DIR / "small_image.jpeg"),
    ]
    list(embedding_model.embed(images))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_id)


@pytest.mark.skip(reason="Requires a multi-gpu server")
@pytest.mark.parametrize("device_ids", [None, [0], [1], [0, 1]])
def test_gpu_cuda_device_ids(device_ids):
    docs = ["hello world", "flag embedding"]
    device_id = device_ids[0] if device_ids else 0
    embedding_model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(
        device_id
    ), f"Text embedding: {options}"

    embedding_model = SparseTextEmbedding(
        "prithvida/Splade_PP_en_v1",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(
        device_id
    ), f"Sparse text embedding: {options}"

    embedding_model = SparseTextEmbedding(
        "Qdrant/bm42-all-minilm-l6-v2-attentions",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(device_id), f"Bm42: {options}"

    embedding_model = LateInteractionTextEmbedding(
        "colbert-ir/colbertv2.0",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(
        device_id
    ), f"Late interaction text embedding: {options}"

    embedding_model = ImageEmbedding(
        model_name="Qdrant/clip-ViT-B-32-vision",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    images = [
        TEST_MISC_DIR / "image.jpeg",
        str(TEST_MISC_DIR / "small_image.jpeg"),
    ]
    list(embedding_model.embed(images))
    options = embedding_model.model.model.get_provider_options()
    assert options["CUDAExecutionProvider"]["device_id"] == str(
        device_id
    ), f"Image embedding: {options}"


@pytest.mark.skip(reason="Requires a multi-gpu server")
@pytest.mark.parametrize(
    "device_ids,parallel", [(None, None), (None, 2), ([1], None), ([1], 1), ([1], 2), ([0, 1], 2)]
)
def test_multi_gpu_parallel_inference(device_ids, parallel):
    docs = ["hello world", "flag embedding"] * 100
    batch_size = 5

    embedding_model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
        lazy_load=True,
    )
    list(embedding_model.embed(docs, batch_size=batch_size, parallel=parallel))

    embedding_model = SparseTextEmbedding(
        "prithvida/Splade_PP_en_v1",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs, batch_size=batch_size, parallel=parallel))

    embedding_model = SparseTextEmbedding(
        "Qdrant/bm42-all-minilm-l6-v2-attentions",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs, batch_size=batch_size, parallel=parallel))

    embedding_model = LateInteractionTextEmbedding(
        "colbert-ir/colbertv2.0",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    list(embedding_model.embed(docs, batch_size=batch_size, parallel=parallel))

    embedding_model = ImageEmbedding(
        model_name="Qdrant/clip-ViT-B-32-vision",
        cuda=True,
        device_ids=device_ids,
        cache_dir=CACHE_DIR,
    )
    images = [
        TEST_MISC_DIR / "image.jpeg",
        str(TEST_MISC_DIR / "small_image.jpeg"),
    ] * 100
    list(embedding_model.embed(images, batch_size=batch_size, parallel=parallel))
