import os
import shutil

import numpy as np
import pytest

from fastembed import MultiTaskTextEmbedding

CANONICAL_VECTOR_VALUES = {
    "jinaai/jina-embeddings-v3": {
        "retrieval.query": {
            "vectors": np.array(
                [
                    [
                        0.07392617,
                        -0.02660709,
                        0.14745504,
                        -0.02088739,
                        0.0789107,
                        -0.09291434,
                        0.00468353,
                        0.04278327,
                        -0.0498863,
                        -0.09133638,
                        -0.02021408,
                        0.01333045,
                        0.09304956,
                        -0.12217413,
                        -0.1517688,
                        -0.06321858,
                        -0.037993,
                        0.00996646,
                        0.00240356,
                        0.01857019,
                        -0.07065693,
                        -0.07262911,
                        0.02474336,
                        0.03223655,
                        0.04860397,
                        0.02118766,
                        0.02885391,
                        -0.06351898,
                        -0.04486841,
                        -0.01556493,
                        0.05945606,
                        -0.04231489,
                    ],
                    [
                        -0.10966361,
                        -0.06452312,
                        0.00457436,
                        0.02245729,
                        0.05487029,
                        -0.01845676,
                        -0.01329387,
                        0.07429793,
                        -0.02465786,
                        -0.01165629,
                        -0.05033085,
                        0.08484261,
                        -0.08245572,
                        0.07067725,
                        -0.07871348,
                        0.0179287,
                        -0.06511015,
                        -0.03859236,
                        -0.05293933,
                        -0.04595497,
                        0.04398658,
                        -0.01258865,
                        -0.05639443,
                        0.04424848,
                        -0.00889678,
                        0.02864096,
                        0.10740122,
                        -0.11583274,
                        -0.04662896,
                        0.03998176,
                        0.06641515,
                        0.01835708,
                    ],
                ]
            ),
            "tolerance": 0.1,
        },
        "retrieval.passage": {
            "vectors": np.array(
                [
                    [
                        -0.03753735,
                        0.05724721,
                        0.07819761,
                        -0.05029617,
                        0.04405113,
                        -0.01000592,
                        0.06124239,
                        -0.05446593,
                        -0.01364476,
                        -0.05984661,
                        -0.02308968,
                        -0.05688294,
                        0.11891583,
                        -0.03224218,
                        -0.01814703,
                        -0.03156558,
                        0.03518531,
                        -0.02014836,
                        -0.02669935,
                        0.0277592,
                        -0.04734779,
                        -0.08216646,
                        0.04565377,
                        -0.02831495,
                        0.0457347,
                        -0.04643558,
                        0.00343047,
                        -0.07974415,
                        0.03208014,
                        -0.05317404,
                        -0.00203198,
                        -0.04074816,
                    ],
                    [
                        -0.14861655,
                        -0.01577728,
                        0.02780116,
                        -0.02535817,
                        0.02535356,
                        0.02791048,
                        0.03155476,
                        0.01918022,
                        0.00426047,
                        -0.03597456,
                        -0.05882818,
                        0.00619552,
                        0.00750015,
                        0.09126588,
                        0.02173683,
                        -0.00374402,
                        -0.01025764,
                        -0.06608114,
                        -0.03461361,
                        -0.00804734,
                        0.01875426,
                        -0.02271635,
                        -0.04308107,
                        0.01970701,
                        0.00303673,
                        -0.03259425,
                        0.08148762,
                        -0.12083711,
                        -0.01994492,
                        0.0050047,
                        0.02426458,
                        -0.00170099,
                    ],
                ]
            ),
            "tolerance": 0.5,
        },
        "separation": {
            "vectors": np.array(
                [
                    [
                        0.09398967,
                        -0.10651711,
                        0.13046436,
                        0.05471611,
                        0.05559288,
                        -0.09735352,
                        -0.0401835,
                        0.11377851,
                        -0.04539866,
                        -0.06424976,
                        -0.01845575,
                        0.1288543,
                        -0.0091803,
                        -0.11028967,
                        -0.14497826,
                        -0.07895527,
                        -0.11136597,
                        0.04933345,
                        0.0073601,
                        0.00208578,
                        -0.07026622,
                        0.03664301,
                        -0.01410522,
                        0.08159963,
                        -0.00312418,
                        0.08604525,
                        0.03245523,
                        0.0243432,
                        -0.1632206,
                        0.02914736,
                        0.12189092,
                        0.01407759,
                    ],
                    [
                        0.03154434,
                        -0.14681207,
                        0.06498899,
                        0.05675182,
                        0.05455389,
                        -0.05962958,
                        -0.05497809,
                        0.14904788,
                        -0.02723428,
                        -0.00792058,
                        -0.03206095,
                        0.17408606,
                        -0.11366311,
                        0.00193246,
                        -0.07406308,
                        -0.02940095,
                        -0.12200059,
                        0.04262922,
                        -0.0104471,
                        -0.04813876,
                        -0.02309961,
                        0.09055562,
                        -0.07385852,
                        0.09457524,
                        -0.03928442,
                        0.08864705,
                        0.04292982,
                        0.00382432,
                        -0.15881643,
                        0.04525036,
                        0.12812722,
                        0.04051839,
                    ],
                ]
            ),
            "tolerance": 1e-6,
        },
        "classification": {
            "vectors": np.array(
                [
                    [
                        0.06063074,
                        -0.08770004,
                        0.1383665,
                        0.00651625,
                        0.07221241,
                        -0.08350758,
                        -0.02725022,
                        0.05653692,
                        -0.0655093,
                        -0.07528743,
                        -0.02392395,
                        0.0541256,
                        0.02816825,
                        -0.12584108,
                        -0.13133372,
                        -0.07734365,
                        -0.07250831,
                        0.01405831,
                        -0.01853171,
                        -0.03641526,
                        -0.03111679,
                        -0.01552055,
                        0.01301841,
                        0.03781024,
                        0.01533106,
                        0.05233759,
                        0.02780087,
                        0.01674484,
                        -0.10845354,
                        -0.00759803,
                        0.09559797,
                        0.01739715,
                    ],
                    [
                        -0.05017639,
                        -0.11898675,
                        0.03198957,
                        0.05141857,
                        0.06892113,
                        -0.03924103,
                        -0.04449044,
                        0.09430395,
                        -0.02638498,
                        -0.00577662,
                        -0.05493132,
                        0.1440273,
                        -0.09812292,
                        0.08394532,
                        -0.06159245,
                        -0.0444326,
                        -0.12405001,
                        -0.02330322,
                        -0.00583696,
                        -0.0561776,
                        -0.00770476,
                        0.03977488,
                        -0.09740996,
                        0.05607887,
                        -0.03055578,
                        0.07926998,
                        0.07461926,
                        0.01560037,
                        -0.08794685,
                        0.02774584,
                        0.04342714,
                        0.0678155,
                    ],
                ]
            ),
            "tolerance": 1e-6,
        },
        "text-matching": {
            "vectors": np.array(
                [
                    [
                        0.09105242,
                        -0.03411407,
                        0.13049866,
                        -0.02600164,
                        0.05764701,
                        -0.11358645,
                        -0.00270602,
                        0.08606345,
                        -0.04159946,
                        -0.08259453,
                        -0.03065768,
                        0.0539673,
                        0.06867454,
                        -0.12614986,
                        -0.16071297,
                        -0.10580365,
                        -0.06129259,
                        0.01145639,
                        0.00125002,
                        0.01299727,
                        -0.05124273,
                        -0.03604639,
                        0.02636702,
                        0.06489752,
                        0.02885577,
                        0.06553873,
                        0.01876015,
                        0.01923829,
                        -0.10008489,
                        -0.0138442,
                        0.0625404,
                        0.00395599,
                    ],
                    [
                        -0.14317708,
                        -0.04998321,
                        0.01329404,
                        0.04642682,
                        0.07889359,
                        0.03539763,
                        -0.00042138,
                        0.05709198,
                        -0.047986,
                        -0.03257386,
                        -0.07362664,
                        0.06120029,
                        -0.08061553,
                        0.06342081,
                        -0.08366095,
                        -0.02656054,
                        -0.05978559,
                        -0.10613462,
                        -0.02092929,
                        -0.02608841,
                        0.01179846,
                        -0.01563349,
                        -0.10146135,
                        0.06646027,
                        -0.03412491,
                        0.08111033,
                        0.13891536,
                        -0.03395027,
                        -0.04849283,
                        -0.00785613,
                        -0.0049023,
                        0.08657855,
                    ],
                ]
            ),
            "tolerance": 1e-6,
        },
    }
}

docs = ["Hello World", "Follow the white rabbit."]


def test_embedding():
    is_ci = os.getenv("CI")

    for model_desc in MultiTaskTextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        dim = model_desc["dim"]

        if isinstance(dim, list):  # if matryoshka
            dim = dim[0]

        model = MultiTaskTextEmbedding(model_name=model_name, cache_dir="models")

        for task in model_desc["tasks"]:
            if task in CANONICAL_VECTOR_VALUES[model_name].keys():
                print(f"evaluating {model_name} task: {task}")

                expected_vectors = CANONICAL_VECTOR_VALUES[model_name][task]["vectors"]
                task_tolerance = CANONICAL_VECTOR_VALUES[model_name][task]["tolerance"]

                embeddings = list(
                    model.task_embed(documents=docs, task_type=task, embeddings_size=dim)
                )
                embedding_arrays = [e.embedding for e in embeddings]
                embeddings = np.stack(embedding_arrays)

                assert np.allclose(embeddings, expected_vectors, atol=task_tolerance)

        if is_ci:
            shutil.rmtree(model.model._model_dir)


def test_batch_embedding():
    is_ci = os.getenv("CI")
    docs_to_embed = docs * 10

    for model_desc in MultiTaskTextEmbedding.list_supported_models():
        if not is_ci and model_desc["size_in_GB"] > 1:
            continue

        model_name = model_desc["model"]
        dim = model_desc["dim"]

        if isinstance(dim, list):  # if matryoshka
            dim = dim[0]

        model = MultiTaskTextEmbedding(model_name=model_name, cache_dir="models")

        for task in model_desc["tasks"]:
            print(f"evaluating {model_name} task: {task}")

            expected_vectors = CANONICAL_VECTOR_VALUES[model_name][task]["vectors"]
            task_tolerance = CANONICAL_VECTOR_VALUES[model_name][task]["tolerance"]

            embeddings = list(
                model.task_embed(
                    documents=docs_to_embed, task_type=task, batch_size=6, embeddings_size=dim
                )
            )
            embedding_arrays = [e.embedding for e in embeddings]
            embeddings = np.stack(embedding_arrays)

            assert np.allclose(embeddings[: len(docs)], expected_vectors, atol=task_tolerance)

        if is_ci:
            shutil.rmtree(model.model._model_dir)


def test_matryoshka_embeddings():
    is_ci = os.getenv("CI")
    embeddings_size = 64

    model = MultiTaskTextEmbedding(model_name="jinaai/jina-embeddings-v3", cache_dir="models")

    with pytest.raises(ValueError):
        embeddings = list(model.task_embed(docs, "text-matching", embeddings_size=100))

    embeddings = list(model.task_embed(docs, "text-matching", embeddings_size=embeddings_size))
    embedding_dimensions = [e.dimension for e in embeddings]

    for dim in embedding_dimensions:
        assert dim == embeddings_size

    if is_ci:
        shutil.rmtree(model.model._model_dir)


def test_parallel_processing():
    is_ci = os.getenv("CI")
    model = MultiTaskTextEmbedding(model_name="jinaai/jina-embeddings-v3")

    token_dim = 1024
    task_type = "text-matching"
    task_tolerance = 1e-3
    docs = ["Hello World", "Follow the white rabbit."] * 100

    embeddings = list(model.task_embed(docs, task_type=task_type, batch_size=10, parallel=2))
    embedding_arrays = [e.embedding for e in embeddings]
    embeddings = np.stack(embedding_arrays)

    embeddings_2 = list(model.task_embed(docs, task_type=task_type, batch_size=10, parallel=None))
    embedding_arrays = [e.embedding for e in embeddings_2]
    embeddings_2 = np.stack(embedding_arrays, axis=0)

    embeddings_3 = list(model.task_embed(docs, task_type=task_type, batch_size=10, parallel=0))
    embedding_arrays = [e.embedding for e in embeddings_3]
    embeddings_3 = np.stack(embedding_arrays, axis=0)

    assert embeddings.shape[0] == len(docs) and embeddings.shape[-1] == token_dim
    assert np.allclose(embeddings, embeddings_2, atol=task_tolerance)
    assert np.allclose(embeddings, embeddings_3, atol=task_tolerance)

    if is_ci:
        shutil.rmtree(model.model._model_dir)


@pytest.mark.parametrize(
    "model_name",
    ["jinaai/jina-embeddings-v3"],
)
def test_lazy_load(model_name):
    model = MultiTaskTextEmbedding(model_name=model_name, cache_dir="models", lazy_load=True)
    assert not hasattr(model.model, "model")

    list(model.task_embed(docs, task_type="text-matching"))
    assert hasattr(model.model, "model")
