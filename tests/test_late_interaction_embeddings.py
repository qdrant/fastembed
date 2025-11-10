import os
from contextlib import contextmanager

import pytest
import numpy as np

from fastembed.late_interaction.late_interaction_text_embedding import (
    LateInteractionTextEmbedding,
)
from tests.utils import delete_model_cache, should_test_model

# vectors are abridged and rounded for brevity
CANONICAL_COLUMN_VALUES = {
    "colbert-ir/colbertv2.0": np.array(
        [
            [0.0759, 0.0841, -0.0299, 0.0374, 0.0254],
            [0.0005, -0.0163, -0.0127, 0.2165, 0.1517],
            [-0.0257, -0.0575, 0.0135, 0.2202, 0.1896],
            [0.0846, 0.0122, 0.0032, -0.0109, -0.1041],
            [0.0477, 0.1078, -0.0314, 0.016, 0.0156],
        ]
    ),
    "answerdotai/answerai-colbert-small-v1": np.array(
        [
            [-0.07281, 0.04632, -0.04711, 0.00762, -0.07374],
            [-0.04464, 0.04426, -0.074, 0.01801, -0.05233],
            [0.09936, -0.05123, -0.04925, -0.05276, -0.08944],
            [0.01644, 0.0203, -0.03789, 0.03165, -0.06501],
            [-0.07281, 0.04633, -0.04711, 0.00762, -0.07374],
        ]
    ),
    "jinaai/jina-colbert-v2": np.array(
        [
            [0.0742, 0.0591, -0.2403, -0.1774, 0.02],
            [0.1318, 0.0882, -0.1138, -0.2066, 0.146],
            [-0.0183, -0.1354, -0.0139, -0.1079, -0.051],
            [0.0003, -0.1184, -0.07, -0.0479, -0.0649],
            [0.0766, 0.0452, -0.2343, -0.183, 0.0058],
        ]
    ),
}

CANONICAL_QUERY_VALUES = {
    "colbert-ir/colbertv2.0": np.array(
        [
            [0.0824, 0.0872, -0.0324, 0.0418, 0.024],
            [-0.0007, -0.0154, -0.0113, 0.2277, 0.1528],
            [-0.0251, -0.0565, 0.0136, 0.2236, 0.1838],
            [0.0848, 0.0056, 0.0041, -0.0036, -0.1032],
            [0.0574, 0.1072, -0.0332, 0.0233, 0.0209],
            [0.1041, 0.0364, -0.0058, -0.027, -0.0704],
            [0.106, 0.0371, -0.0055, -0.0339, -0.0719],
            [0.1063, 0.0363, 0.0014, -0.0334, -0.0698],
            [0.112, 0.036, 0.0026, -0.0355, -0.0675],
            [0.1184, 0.0441, 0.0166, -0.0169, -0.0244],
            [0.1033, 0.035, 0.0183, 0.0475, 0.0612],
            [-0.0028, -0.014, -0.016, 0.2175, 0.1537],
            [0.0547, 0.0219, -0.007, 0.1748, 0.1154],
            [-0.001, -0.0184, -0.0112, 0.2197, 0.1523],
            [-0.0012, -0.0149, -0.0119, 0.2147, 0.152],
            [-0.0186, -0.0239, -0.014, 0.2196, 0.156],
            [-0.017, -0.0232, -0.0108, 0.2212, 0.157],
            [-0.0109, -0.0024, -0.003, 0.1972, 0.1391],
            [0.0898, 0.0219, -0.0255, 0.0734, -0.0096],
            [0.1143, 0.015, -0.022, 0.0417, -0.0421],
            [0.1056, 0.0091, -0.0137, 0.0129, -0.0619],
            [0.0234, 0.004, -0.0285, 0.1565, 0.0883],
            [-0.0037, -0.0079, -0.0204, 0.1982, 0.1502],
            [0.0988, 0.0377, 0.0226, 0.0309, 0.0508],
            [-0.0103, -0.0128, -0.0035, 0.2114, 0.155],
            [-0.0103, -0.0184, -0.011, 0.2252, 0.157],
            [-0.0033, -0.0292, -0.0097, 0.2237, 0.1607],
            [-0.0198, -0.0257, -0.0193, 0.2265, 0.165],
            [-0.0227, -0.0028, -0.0084, 0.1995, 0.1306],
            [0.0916, 0.0185, -0.0186, 0.0173, -0.0577],
            [0.1022, 0.0228, -0.0174, -0.0102, -0.065],
            [0.1043, 0.0231, -0.0144, -0.0246, -0.067],
        ]
    ),
    "answerdotai/answerai-colbert-small-v1": np.array(
        [
            [-0.07284, 0.04657, -0.04746, 0.00786, -0.07342],
            [-0.0473, 0.04615, -0.07551, 0.01591, -0.0517],
            [0.09658, -0.0506, -0.04593, -0.05225, -0.09086],
            [0.01815, 0.0165, -0.03366, 0.03214, -0.07019],
            [-0.07284, 0.04657, -0.04746, 0.00787, -0.07342],
            [-0.07748, 0.04493, -0.055, 0.00481, -0.0486],
            [-0.0803, 0.04229, -0.0589, 0.00379, -0.04506],
            [-0.08477, 0.03724, -0.06162, 0.00578, -0.04554],
            [-0.08392, 0.03805, -0.06202, 0.00899, -0.0409],
            [-0.07945, 0.04163, -0.06151, 0.00569, -0.04432],
            [-0.08469, 0.03985, -0.05765, 0.00485, -0.04485],
            [-0.08306, 0.04111, -0.05774, 0.00583, -0.04325],
            [-0.08244, 0.04597, -0.05842, 0.00433, -0.04025],
            [-0.08385, 0.04745, -0.05845, 0.00469, -0.04002],
            [-0.08402, 0.05014, -0.05941, 0.00692, -0.03452],
            [-0.08303, 0.05693, -0.05701, 0.00504, -0.03565],
            [-0.08216, 0.05516, -0.05687, 0.0057, -0.03748],
            [-0.08051, 0.05751, -0.05647, 0.00283, -0.03645],
            [-0.08172, 0.05608, -0.06064, 0.00252, -0.03533],
            [-0.08073, 0.06144, -0.06373, 0.00935, -0.03154],
            [-0.06651, 0.06697, -0.06769, 0.01717, -0.03369],
            [-0.06526, 0.06931, -0.06935, 0.0139, -0.03702],
            [-0.05435, 0.05829, -0.06593, 0.01708, -0.04559],
            [-0.03648, 0.05234, -0.06759, 0.02057, -0.05053],
            [-0.03461, 0.05032, -0.06747, 0.02216, -0.05209],
            [-0.03444, 0.04835, -0.06812, 0.02296, -0.05276],
            [-0.03292, 0.04853, -0.06811, 0.02348, -0.05303],
            [-0.03349, 0.04783, -0.06846, 0.02393, -0.05334],
            [-0.03485, 0.04677, -0.06826, 0.02362, -0.05326],
            [-0.03408, 0.04744, -0.06931, 0.02302, -0.05288],
            [-0.03444, 0.04838, -0.06945, 0.02133, -0.05277],
            [-0.03473, 0.04792, -0.07033, 0.02196, -0.05314],
        ]
    ),
    "jinaai/jina-colbert-v2": np.array(
        [
            [0.0477, 0.0255, -0.2224, -0.1085, -0.03],
            [0.0206, -0.0845, -0.0075, -0.1712, 0.0156],
            [-0.0056, -0.0957, -0.0147, -0.1277, -0.0225],
            [0.0486, -0.0499, -0.1609, 0.0194, 0.0274],
            [0.0481, 0.0253, -0.2278, -0.1126, -0.0294],
            [0.0599, -0.0678, -0.0956, -0.0757, 0.0236],
            [0.0592, -0.0862, -0.0621, -0.1084, 0.0155],
            [0.0874, -0.0714, -0.0772, -0.1414, 0.037],
            [0.1009, -0.0552, -0.0669, -0.163, 0.0493],
            [0.1135, -0.047, -0.0576, -0.1699, 0.0538],
            [0.1228, -0.0428, -0.0507, -0.1725, 0.0562],
            [0.1291, -0.0388, -0.042, -0.1753, 0.0569],
            [0.1365, -0.0337, -0.0326, -0.1786, 0.0574],
            [0.1439, -0.026, -0.024, -0.1831, 0.0574],
            [0.1527, -0.0099, -0.0179, -0.1874, 0.057],
            [0.1555, 0.0186, -0.023, -0.1801, 0.0539],
            [0.1389, 0.054, -0.0345, -0.1636, 0.0429],
            [0.1058, 0.0862, -0.0418, -0.1455, 0.0222],
            [0.0713, 0.1061, -0.0438, -0.1288, 0.0002],
            [0.0453, 0.1143, -0.0457, -0.1119, -0.019],
            [0.0346, 0.1131, -0.0487, -0.0952, -0.0338],
            [0.0355, 0.1073, -0.0493, -0.0823, -0.0438],
            [0.0424, 0.1041, -0.0459, -0.0761, -0.048],
            [0.048, 0.102, -0.0421, -0.0718, -0.0477],
            [0.0474, 0.0989, -0.0413, -0.0654, -0.0431],
            [0.0434, 0.095, -0.0415, -0.0589, -0.0345],
            [0.0408, 0.0897, -0.0405, -0.0554, -0.0197],
            [0.0433, 0.0811, -0.0407, -0.0545, 0.0055],
            [0.0514, 0.0629, -0.0446, -0.0549, 0.0368],
            [0.058, 0.048, -0.0527, -0.0607, 0.0568],
            [0.0561, 0.0447, -0.0661, -0.0702, 0.0764],
            [0.0204, -0.0856, -0.0386, -0.1232, -0.0332],
        ]
    ),
}

MODELS_TO_CACHE = ("answerdotai/answerai-colbert-small-v1",)


@pytest.fixture(scope="module")
def model_cache():
    is_ci = os.getenv("CI")
    cache = {}

    @contextmanager
    def get_model(model_name: str):
        if model_name not in cache:
            cache[model_name] = LateInteractionTextEmbedding(model_name)
        yield cache[model_name]
        if model_name not in MODELS_TO_CACHE:
            model_inst = cache.pop(model_name)
            if is_ci:
                delete_model_cache(model_inst.model._model_dir)
            del model_inst

    yield get_model

    if is_ci:
        for name, model in cache.items():
            delete_model_cache(model.model._model_dir)
    cache.clear()


docs = ["Hello World"]


@pytest.mark.parametrize("model_name", ["answerdotai/answerai-colbert-small-v1"])
def test_batch_embedding(model_cache, model_name: str):
    docs_to_embed = docs * 10

    with model_cache(model_name) as model:
        result = list(model.embed(docs_to_embed, batch_size=6))
        expected_result = CANONICAL_COLUMN_VALUES[model_name]

        for value in result:
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(value[:, :abridged_dim], expected_result, atol=2e-3)


@pytest.mark.parametrize("model_name", ["answerdotai/answerai-colbert-small-v1"])
def test_batch_inference_size_same_as_single_inference(model_cache, model_name: str):
    with model_cache(model_name) as model:
        docs_to_embed = [
            "short document",
            "A bit longer document, which should not affect the size",
        ]
        result = list(model.embed(docs_to_embed, batch_size=1))
        result_2 = list(model.embed(docs_to_embed, batch_size=2))
        assert len(result[0]) == len(result_2[0])


@pytest.mark.parametrize("model_name", ["answerdotai/answerai-colbert-small-v1"])
def test_single_embedding(model_cache, model_name: str):
    is_ci = os.getenv("CI")
    is_manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
    docs_to_embed = docs

    for model_desc in LateInteractionTextEmbedding._list_supported_models():
        if not should_test_model(model_desc, model_name, is_ci, is_manual):
            continue

        print("evaluating", model_name)
        with model_cache(model_desc.model) as model:
            whole_result = list(model.embed(docs_to_embed, batch_size=6))
            assert len(whole_result) == 1
            result = whole_result[0]
            expected_result = CANONICAL_COLUMN_VALUES[model_desc.model]
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(result[:, :abridged_dim], expected_result, atol=2e-3)


@pytest.mark.parametrize("model_name", ["answerdotai/answerai-colbert-small-v1"])
def test_single_embedding_query(model_cache, model_name: str):
    is_ci = os.getenv("CI")
    is_manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
    queries_to_embed = docs

    for model_desc in LateInteractionTextEmbedding._list_supported_models():
        if not should_test_model(model_desc, model_name, is_ci, is_manual):
            continue

        print("evaluating", model_desc.model)
        with model_cache(model_desc.model) as model:
            whole_result = list(model.query_embed(queries_to_embed))
            assert len(whole_result) == 1
            result = whole_result[0]
            expected_result = CANONICAL_QUERY_VALUES[model_desc.model]
            token_num, abridged_dim = expected_result.shape
            assert np.allclose(result[:, :abridged_dim], expected_result, atol=2e-3)


@pytest.mark.parametrize("token_dim,model_name", [(96, "answerdotai/answerai-colbert-small-v1")])
def test_parallel_processing(model_cache, token_dim: int, model_name: str):
    # this test loads a copy of a model per process, might cause oom in parallel=0 on machines with
    # an insufficient mem-to-cpus-ratio
    with model_cache(model_name) as model:
        docs = ["hello world", "flag embedding"] * 100
        embeddings = list(model.embed(docs, batch_size=10, parallel=2))

        embeddings_2 = list(model.embed(docs, batch_size=10, parallel=None))

        embeddings_3 = list(model.embed(docs, batch_size=10, parallel=0))

        assert len(embeddings) == len(docs) and embeddings[0].shape[-1] == token_dim

        for i in range(len(embeddings)):
            assert np.allclose(embeddings[i], embeddings_2[i], atol=1e-3)
            assert np.allclose(embeddings[i], embeddings_3[i], atol=1e-3)


@pytest.mark.parametrize("model_name", ["answerdotai/answerai-colbert-small-v1"])
def test_lazy_load(model_name: str):
    is_ci = os.getenv("CI")

    model = LateInteractionTextEmbedding(model_name=model_name, lazy_load=True)
    assert not hasattr(model.model, "model")

    docs = ["hello world", "flag embedding"]
    list(model.embed(docs))
    assert hasattr(model.model, "model")

    model = LateInteractionTextEmbedding(model_name=model_name, lazy_load=True)
    list(model.query_embed(docs))

    model = LateInteractionTextEmbedding(model_name=model_name, lazy_load=True)
    list(model.passage_embed(docs))

    if is_ci:
        delete_model_cache(model.model._model_dir)


def test_get_embedding_size():
    model_name = "answerdotai/answerai-colbert-small-v1"
    assert LateInteractionTextEmbedding.get_embedding_size(model_name) == 96

    model_name = "answerdotai/answerai-ColBERT-small-v1"
    assert LateInteractionTextEmbedding.get_embedding_size(model_name) == 96


def test_embedding_size():
    is_ci = os.getenv("CI")
    model_name = "answerdotai/answerai-colbert-small-v1"
    model = LateInteractionTextEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 96

    model_name = "answerdotai/answerai-ColBERT-small-v1"
    model = LateInteractionTextEmbedding(model_name=model_name, lazy_load=True)
    assert model.embedding_size == 96
    if is_ci:
        delete_model_cache(model.model._model_dir)
