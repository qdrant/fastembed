import itertools
import os
import numpy as np
import pytest

from fastembed.common.model_description import (
    PoolingType,
    ModelSource,
    DenseModelDescription,
    BaseModelDescription,
)
from fastembed.common.onnx_model import OnnxOutputContext
from fastembed.common.utils import normalize, mean_pooling
from fastembed.text.custom_text_embedding import CustomTextEmbedding, PostprocessingConfig
from fastembed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.text.text_embedding import TextEmbedding
from tests.utils import delete_model_cache


@pytest.fixture(autouse=True)
def restore_custom_models_fixture():
    CustomTextEmbedding.SUPPORTED_MODELS = []
    CustomTextCrossEncoder.SUPPORTED_MODELS = []
    yield
    CustomTextEmbedding.SUPPORTED_MODELS = []
    CustomTextCrossEncoder.SUPPORTED_MODELS = []


def test_text_custom_model():
    is_ci = os.getenv("CI")
    custom_model_name = "intfloat/multilingual-e5-small"
    canonical_vector = np.array(
        [3.1317e-02, 3.0939e-02, -3.5117e-02, -6.7274e-02, 8.5084e-02], dtype=np.float32
    )
    pooling = PoolingType.MEAN
    normalization = True
    dim = 384
    size_in_gb = 0.47
    source = ModelSource(hf=custom_model_name)

    TextEmbedding.add_custom_model(
        custom_model_name,
        pooling=pooling,
        normalization=normalization,
        sources=source,
        dim=dim,
        size_in_gb=size_in_gb,
    )

    assert CustomTextEmbedding.SUPPORTED_MODELS[0] == DenseModelDescription(
        model=custom_model_name,
        sources=source,
        model_file="onnx/model.onnx",
        description="",
        license="",
        size_in_GB=size_in_gb,
        additional_files=[],
        dim=dim,
        tasks={},
    )
    assert CustomTextEmbedding.POSTPROCESSING_MAPPING[custom_model_name] == PostprocessingConfig(
        pooling=pooling, normalization=normalization
    )

    model = TextEmbedding(custom_model_name)
    docs = ["hello world", "flag embedding"]
    embeddings = list(model.embed(docs))
    embeddings = np.stack(embeddings, axis=0)
    assert embeddings.shape == (2, dim)

    assert np.allclose(embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3)

    if is_ci:
        delete_model_cache(model.model._model_dir)

    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()


def test_cross_encoder_custom_model():
    is_ci = os.getenv("CI")
    custom_model_name = "Xenova/ms-marco-MiniLM-L-4-v2"
    size_in_gb = 0.08
    source = ModelSource(hf=custom_model_name)
    canonical_vector = np.array([-5.7170815, -11.112114], dtype=np.float32)

    TextCrossEncoder.add_custom_model(
        custom_model_name,
        model_file="onnx/model.onnx",
        sources=source,
        size_in_gb=size_in_gb,
    )

    assert CustomTextCrossEncoder.SUPPORTED_MODELS[0] == BaseModelDescription(
        model=custom_model_name,
        sources=source,
        model_file="onnx/model.onnx",
        description="",
        license="",
        size_in_GB=size_in_gb,
    )

    model = TextCrossEncoder(custom_model_name)
    pairs = [
        ("What is AI?", "Artificial intelligence is ..."),
        ("What is ML?", "Machine learning is ..."),
    ]
    scores = list(model.rerank_pairs(pairs))

    embeddings = np.stack(scores, axis=0)
    assert embeddings.shape == (2,)
    assert np.allclose(embeddings, canonical_vector, atol=1e-3)
    if is_ci:
        delete_model_cache(model.model._model_dir)

    CustomTextCrossEncoder.SUPPORTED_MODELS.clear()


def test_mock_add_custom_models():
    dim = 5
    size_in_gb = 0.1
    source = ModelSource(hf="artificial")

    num_tokens = 10
    dummy_pooled_embedding = np.random.random((1, dim)).astype(np.float32)
    dummy_token_embedding = np.random.random((1, num_tokens, dim)).astype(np.float32)
    dummy_attention_mask = np.ones((1, num_tokens)).astype(np.int64)

    dummy_token_output = OnnxOutputContext(
        model_output=dummy_token_embedding, attention_mask=dummy_attention_mask
    )
    dummy_pooled_output = OnnxOutputContext(model_output=dummy_pooled_embedding)
    input_data = {
        f"{PoolingType.MEAN.lower()}-normalized": dummy_token_output,
        f"{PoolingType.MEAN.lower()}": dummy_token_output,
        f"{PoolingType.CLS.lower()}-normalized": dummy_token_output,
        f"{PoolingType.CLS.lower()}": dummy_token_output,
        f"{PoolingType.DISABLED.lower()}-normalized": dummy_pooled_output,
        f"{PoolingType.DISABLED.lower()}": dummy_pooled_output,
    }

    expected_output = {
        f"{PoolingType.MEAN.lower()}-normalized": normalize(
            mean_pooling(dummy_token_embedding, dummy_attention_mask)
        ),
        f"{PoolingType.MEAN.lower()}": mean_pooling(dummy_token_embedding, dummy_attention_mask),
        f"{PoolingType.CLS.lower()}-normalized": normalize(dummy_token_embedding[:, 0]),
        f"{PoolingType.CLS.lower()}": dummy_token_embedding[:, 0],
        f"{PoolingType.DISABLED.lower()}-normalized": normalize(dummy_pooled_embedding),
        f"{PoolingType.DISABLED.lower()}": dummy_pooled_embedding,
    }

    for pooling, normalization in itertools.product(
        (PoolingType.MEAN, PoolingType.CLS, PoolingType.DISABLED), (True, False)
    ):
        model_name = f"{pooling.name.lower()}{'-normalized' if normalization else ''}"
        TextEmbedding.add_custom_model(
            model_name,
            pooling=pooling,
            normalization=normalization,
            sources=source,
            dim=dim,
            size_in_gb=size_in_gb,
        )

        custom_text_embedding = CustomTextEmbedding(
            model_name,
            lazy_load=True,
            specific_model_path="./",  # disable model downloading and loading
        )

        post_processed_output = next(
            iter(custom_text_embedding._post_process_onnx_output(input_data[model_name]))
        )
        assert np.allclose(post_processed_output, expected_output[model_name], atol=1e-3)

    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()


def test_do_not_add_existing_model():
    existing_base_model = "sentence-transformers/all-MiniLM-L6-v2"
    custom_model_name = "intfloat/multilingual-e5-small"

    with pytest.raises(ValueError, match=f"Model {existing_base_model} is already registered"):
        TextEmbedding.add_custom_model(
            existing_base_model,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=existing_base_model),
            dim=384,
            size_in_gb=0.47,
        )

    TextEmbedding.add_custom_model(
        custom_model_name,
        pooling=PoolingType.MEAN,
        normalization=False,
        sources=ModelSource(hf=existing_base_model),
        dim=384,
        size_in_gb=0.47,
    )

    with pytest.raises(ValueError, match=f"Model {custom_model_name} is already registered"):
        TextEmbedding.add_custom_model(
            custom_model_name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=custom_model_name),
            dim=384,
            size_in_gb=0.47,
        )

    CustomTextEmbedding.SUPPORTED_MODELS.clear()
    CustomTextEmbedding.POSTPROCESSING_MAPPING.clear()


def test_do_not_add_existing_cross_encoder():
    existing_base_model = "Xenova/ms-marco-MiniLM-L-6-v2"
    custom_model_name = "Xenova/ms-marco-MiniLM-L-4-v2"

    with pytest.raises(ValueError, match=f"Model {existing_base_model} is already registered"):
        TextCrossEncoder.add_custom_model(
            existing_base_model,
            sources=ModelSource(hf=existing_base_model),
            size_in_gb=0.08,
        )

    TextCrossEncoder.add_custom_model(
        custom_model_name,
        sources=ModelSource(hf=existing_base_model),
        size_in_gb=0.08,
    )

    with pytest.raises(ValueError, match=f"Model {custom_model_name} is already registered"):
        TextCrossEncoder.add_custom_model(
            custom_model_name,
            sources=ModelSource(hf=custom_model_name),
            size_in_gb=0.08,
        )

    CustomTextCrossEncoder.SUPPORTED_MODELS.clear()
