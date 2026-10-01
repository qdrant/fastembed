import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from tokenizers import Tokenizer, models, pre_tokenizers

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from fastembed.text.custom_text_embedding import CustomTextEmbedding


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    monkeypatch.setattr(CustomTextEmbedding, "SUPPORTED_MODELS", [])
    monkeypatch.setattr(CustomTextEmbedding, "POSTPROCESSING_MAPPING", {})
    tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "hello": 2}))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 16, "pad_token": "[PAD]"})
    )
    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 0}))
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input_ids"], ["first"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["first", "offset"], ["sentence_embedding"]),
            helper.make_node("Unsqueeze", ["sentence_embedding", "axis"], ["hidden_states"]),
        ],
        "two_outputs",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
        [
            helper.make_tensor_value_info("first", TensorProto.FLOAT, ["batch", "seq"]),
            helper.make_tensor_value_info(
                "sentence_embedding", TensorProto.FLOAT, ["batch", "seq"]
            ),
            helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch", "seq", 1]),
        ],
        [
            helper.make_tensor("offset", TensorProto.FLOAT, [1], [10.0]),
            helper.make_tensor("axis", TensorProto.INT64, [1], [2]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, tmp_path / "model.onnx")
    return tmp_path


@pytest.mark.parametrize("parallel", [None, 2])
@pytest.mark.parametrize("lazy_load", [False, True])
@pytest.mark.parametrize("output_name,expected", [(None, 2.0), ("sentence_embedding", 12.0)])
def test_output_selection_in_real_onnx_session(
    local_model, parallel, lazy_load, output_name, expected
):
    TextEmbedding.add_custom_model(
        "local/output-test",
        PoolingType.DISABLED,
        False,
        ModelSource(hf="local/output-test"),
        dim=1,
        model_file="model.onnx",
        output_name=output_name,
    )
    model = TextEmbedding(
        "local/output-test",
        specific_model_path=str(local_model),
        cache_dir=str(local_model),
        lazy_load=lazy_load,
        threads=1,
        cuda=False,
    )
    vectors = np.stack(list(model.embed(["hello"] * 4, batch_size=2, parallel=parallel)))
    np.testing.assert_array_equal(vectors, np.full((4, 1), expected, dtype=np.float32))


def test_unknown_output_name_raises_from_onnx(local_model):
    TextEmbedding.add_custom_model(
        "local/missing",
        PoolingType.DISABLED,
        False,
        ModelSource(hf="local/missing"),
        dim=1,
        model_file="model.onnx",
        output_name="missing",
    )
    model = TextEmbedding(
        "local/missing",
        specific_model_path=str(local_model),
        cache_dir=str(local_model),
        cuda=False,
    )
    with pytest.raises(InvalidArgument, match="Invalid output name"):
        list(model.embed("hello"))


@pytest.mark.parametrize("output_name", ["", "   ", 123, []])
def test_invalid_output_name_does_not_register_model(local_model, output_name):
    with pytest.raises(ValueError, match="output_name"):
        TextEmbedding.add_custom_model(
            "local/invalid",
            PoolingType.DISABLED,
            False,
            ModelSource(hf="local/invalid"),
            dim=1,
            output_name=output_name,
        )
    assert CustomTextEmbedding.SUPPORTED_MODELS == []
    assert CustomTextEmbedding.POSTPROCESSING_MAPPING == {}


@pytest.mark.parametrize("pooling", [PoolingType.CLS, PoolingType.MEAN, PoolingType.LAST_TOKEN])
@pytest.mark.parametrize("normalization", [False, True])
def test_selected_output_uses_registered_postprocessing(local_model, pooling, normalization):
    TextEmbedding.add_custom_model(
        "local/pooled",
        pooling,
        normalization,
        ModelSource(hf="local/pooled"),
        dim=1,
        model_file="model.onnx",
        output_name="hidden_states",
    )
    model = TextEmbedding(
        "local/pooled",
        specific_model_path=str(local_model),
        cache_dir=str(local_model),
        cuda=False,
    )
    expected = 1.0 if normalization else 12.0
    np.testing.assert_allclose(
        list(model.embed(["hello", "hello hello"])), [[expected], [expected]]
    )
