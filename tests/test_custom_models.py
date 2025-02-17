import os
import numpy as np
import pytest
from dataclasses import replace

from fastembed.common.model_description import PoolingType, ModelSource, DenseModelDescription
from fastembed.text.clip_embedding import CLIPOnnxEmbedding
from fastembed.text.multitask_embedding import JinaEmbeddingV3
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed.text.pooled_embedding import PooledEmbedding
from fastembed.text.text_embedding import TextEmbedding, PooledNormalizedEmbedding
from tests.utils import delete_model_cache


def restore_custom_models():
    for embedding_cls in TextEmbedding.EMBEDDINGS_REGISTRY:
        assert hasattr(embedding_cls, "CUSTOM_MODELS")
        embedding_cls.CUSTOM_MODELS = []


@pytest.fixture(autouse=True)
def restore_custom_models_fixture():
    restore_custom_models()
    yield
    restore_custom_models()


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

    assert PooledNormalizedEmbedding.CUSTOM_MODELS[0] == DenseModelDescription(
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

    model = TextEmbedding(custom_model_name)
    docs = ["hello world", "flag embedding"]
    embeddings = list(model.embed(docs))
    embeddings = np.stack(embeddings, axis=0)
    assert embeddings.shape == (2, dim)

    assert np.allclose(embeddings[0, : canonical_vector.shape[0]], canonical_vector, atol=1e-3)
    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_mock_add_custom_models():
    def check_custom_models_number(cls_to_num_map):
        for embed_cls, num_models in cls_to_num_map.items():
            assert len(embed_cls.CUSTOM_MODELS) == num_models

    custom_model_name = "intfloat/multilingual-e5-small"
    dim = 384
    size_in_gb = 0.47
    source = ModelSource(hf=custom_model_name)
    role_model_description = DenseModelDescription(
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
    current_supported_models_number = len(TextEmbedding._list_supported_models())

    class_num_models_map = {
        PooledNormalizedEmbedding: 0,
        OnnxTextEmbedding: 0,
        PooledEmbedding: 0,
        CLIPOnnxEmbedding: 0,
        JinaEmbeddingV3: 0,
    }

    check_custom_models_number(class_num_models_map)
    TextEmbedding.add_custom_model(
        f"{custom_model_name}-mean-normalize",
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=source,
        dim=dim,
        size_in_gb=size_in_gb,
    )
    assert PooledNormalizedEmbedding.CUSTOM_MODELS[0] == replace(
        role_model_description, model=f"{custom_model_name}-mean-normalize"
    )
    class_num_models_map[PooledNormalizedEmbedding] += 1
    check_custom_models_number(class_num_models_map)
    current_supported_models_number += 1
    assert len(TextEmbedding._list_supported_models()) == current_supported_models_number

    TextEmbedding.add_custom_model(
        f"{custom_model_name}-cls-no-normalize",
        pooling=PoolingType.CLS,
        normalization=True,
        sources=source,
        dim=dim,
        size_in_gb=size_in_gb,
    )
    assert OnnxTextEmbedding.CUSTOM_MODELS[0] == replace(
        role_model_description, model=f"{custom_model_name}-cls-no-normalize"
    )
    class_num_models_map[OnnxTextEmbedding] += 1
    check_custom_models_number(class_num_models_map)
    current_supported_models_number += 1
    assert len(TextEmbedding._list_supported_models()) == current_supported_models_number

    TextEmbedding.add_custom_model(
        f"{custom_model_name}-no-pooling-normalize",
        pooling=PoolingType.DISABLED,
        normalization=True,
        sources=source,
        dim=dim,
        size_in_gb=size_in_gb,
    )
    assert OnnxTextEmbedding.CUSTOM_MODELS[1] == replace(
        role_model_description, model=f"{custom_model_name}-no-pooling-normalize"
    )
    class_num_models_map[OnnxTextEmbedding] += 1
    check_custom_models_number(class_num_models_map)
    current_supported_models_number += 1
    assert len(TextEmbedding._list_supported_models()) == current_supported_models_number

    TextEmbedding.add_custom_model(
        f"{custom_model_name}-mean-no-normalize",
        pooling=PoolingType.MEAN,
        normalization=False,
        sources=source,
        dim=dim,
        size_in_gb=size_in_gb,
    )
    assert PooledEmbedding.CUSTOM_MODELS[0] == replace(
        role_model_description, model=f"{custom_model_name}-mean-no-normalize"
    )
    class_num_models_map[PooledEmbedding] += 1
    check_custom_models_number(class_num_models_map)
    current_supported_models_number += 1
    assert len(TextEmbedding._list_supported_models()) == current_supported_models_number

    TextEmbedding.add_custom_model(
        f"{custom_model_name}-no-pooling-no-normalize",
        pooling=PoolingType.DISABLED,
        normalization=False,
        sources=source,
        dim=dim,
        size_in_gb=size_in_gb,
    )
    assert CLIPOnnxEmbedding.CUSTOM_MODELS[0] == replace(
        role_model_description, model=f"{custom_model_name}-no-pooling-no-normalize"
    )
    class_num_models_map[CLIPOnnxEmbedding] += 1
    check_custom_models_number(class_num_models_map)
    current_supported_models_number += 1
    assert len(TextEmbedding._list_supported_models()) == current_supported_models_number

    TextEmbedding.add_custom_model(
        f"{custom_model_name}-mean-normalize-multitask",
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=source,
        dim=dim,
        size_in_gb=size_in_gb,
        tasks={"task1": 1},
    )
    assert JinaEmbeddingV3.CUSTOM_MODELS[0] == replace(
        role_model_description,
        tasks={"task1": 1},
        model=f"{custom_model_name}-mean-normalize-multitask",
    )
    class_num_models_map[JinaEmbeddingV3] += 1
    check_custom_models_number(class_num_models_map)
    current_supported_models_number += 1
    assert len(TextEmbedding._list_supported_models()) == current_supported_models_number


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
