# ⚡️ What is FastEmbed?

FastEmbed is a lightweight, fast, Python library built for embedding generation. We [support popular text models](https://qdrant.github.io/fastembed/examples/Supported_Models/). Please [open a GitHub issue](https://github.com/qdrant/fastembed/issues/new) if you want us to add a new model.

The default text embedding (`TextEmbedding`) model is Flag Embedding, presented in the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard. It supports "query" and "passage" prefixes for the input text. Here is an example for [Retrieval Embedding Generation](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/) and how to use [FastEmbed with Qdrant](https://qdrant.github.io/fastembed/qdrant/Usage_With_Qdrant/).

## 📈 Why FastEmbed?

1. Light: FastEmbed is a lightweight library with few external dependencies. We don't require a GPU and don't download GBs of PyTorch dependencies, and instead use the ONNX Runtime. This makes it a great candidate for serverless runtimes like AWS Lambda. 

2. Fast: FastEmbed is designed for speed. We use the ONNX Runtime, which is faster than PyTorch. We also use data parallelism for encoding large datasets.

3. Accurate: FastEmbed is better than OpenAI Ada-002. We also [support](https://qdrant.github.io/fastembed/examples/Supported_Models/) an ever-expanding set of models, including a few multilingual models.

## 🚀 Installation

To install the FastEmbed library, pip works best. You can install it with or without GPU support:

```bash
pip install fastembed

# or with GPU support

pip install fastembed-gpu
```

## Development of this fork

This fork is based on [Qdrant FastEmbed](https://github.com/qdrant/fastembed) at
`d076f083519746d132888079e6e790f2193c9c08` (FastEmbed 0.9.0). It preserves
the upstream architecture, public API, dependency declarations, `poetry.lock`,
`LICENSE`, and `NOTICE`.

### Install and test the source

Python 3.10 or newer is required; this change was tested on Python 3.11.15 on
Windows using Poetry 2.5.1 and the existing dependency lockfile.

```bash
git clone https://github.com/irulappan151204/fastembed.git
cd fastembed
git switch fix/model-download-resources
python -m pip install "poetry==2.5.1"
poetry install --only main,test,types
poetry run pytest tests/test_gcs_download.py tests/test_model_management.py tests/test_common.py tests/test_image_transform.py tests/test_parallel_processor.py
poetry run mypy fastembed --disallow-incomplete-defs --disallow-untyped-defs --disable-error-code=import-untyped
```

For linting, the upstream pre-commit hooks pin Ruff 0.3.4. With `uv` installed:

```bash
uv tool run --from "ruff==0.3.4" ruff check --config pyproject.toml fastembed/common/model_management.py tests/test_gcs_download.py
uv tool run --from "ruff==0.3.4" ruff format --check --config pyproject.toml fastembed/common/model_management.py tests/test_gcs_download.py
```

The tests above use local HTTP fixtures and require no model weights, GPU, API
keys, paid services, or external model endpoints. The full suite is
`poetry run pytest`; it includes model downloads and hardware-dependent tests.
To run the representative reranker tests using CPU inference and the upstream
CI model-selection policy (which also cleans up downloaded test models):

```bash
CI=1 poetry run pytest tests/test_text_cross_encoder.py
```

### Environment configuration

FastEmbed accepts constructor arguments such as `cache_dir`, `threads`,
`providers`, and `local_files_only`. The following environment variables are
also useful:

| Variable | Purpose |
| --- | --- |
| `FASTEMBED_CACHE_PATH` | Choose a writable directory for downloaded models. |
| `HF_HUB_OFFLINE=1` | Require cached models; uncached inference tests will fail. |
| `HF_ENDPOINT` | Select a trusted Hugging Face mirror if required. |
| `HF_TOKEN` | Optional authentication for restricted models; set privately and never commit it. |

For example, in PowerShell:

```powershell
$env:FASTEMBED_CACHE_PATH = Join-Path $env:TEMP "fastembed-models"
poetry run pytest tests/test_text_cross_encoder.py
```

In restricted Windows environments, a missing `PROCESSOR_ARCHITECTURE` can
cause Poetry to report `Could not parse version constraint`. Set that variable
to the actual architecture for the current process (`AMD64` on the tested
64-bit Windows host). If Documents disallows tool writes, put the checkout and
virtual environment in a writable development directory, or redirect tool
caches to a writable directory. `pytest -p no:cacheprovider` and Ruff's
`--no-cache` avoid writing their caches into the checkout.

### Fixes in this fork

- Streamed model-download responses now close on success and failure, including
  HTTP errors, malformed length headers, and destination-open errors. Existing
  exception types, timeouts, chunk sizes, and return values are preserved.
- The missing-length warning, fallback-download log, and missing-model-directory
  error omit custom source URLs, which may contain signed access tokens. The
  warning uses the existing Loguru logger and still explains disabled progress.
- Nine regression and compatibility tests use a real local HTTP server to
  verify resource cleanup, unchanged file bytes, and safe diagnostics.

These changes do not sanitize every third-party exception; callers should avoid
logging raw HTTP exceptions or sensitive configuration. Model weights have
their own licenses; see `NOTICE` and each model's license before using them.

## 📖 Quickstart

```python
from fastembed import TextEmbedding


# Example list of documents
documents: list[str] = [
    "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
    "fastembed is supported by and maintained by Qdrant.",
]

# This will trigger the model download and initialization
embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

embeddings_generator = embedding_model.embed(documents)  # reminder this is a generator
embeddings_list = list(embedding_model.embed(documents))
  # you can also convert the generator to a list, and that to a numpy array
len(embeddings_list[0]) # Vector of 384 dimensions
```

Fastembed supports a variety of models for different tasks and modalities.
The list of all the available models can be found [here](https://qdrant.github.io/fastembed/examples/Supported_Models/)
### 🎒 Dense text embeddings

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(documents))

# [
#   array([-0.1115,  0.0097,  0.0052,  0.0195, ...], dtype=float32),
#   array([-0.1019,  0.0635, -0.0332,  0.0522, ...], dtype=float32)
# ]

```

Dense text embedding can also be extended with models which are not in the list of supported models.

```python
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

TextEmbedding.add_custom_model(
    model="intfloat/multilingual-e5-small",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="intfloat/multilingual-e5-small"),  # can be used with an `url` to load files from a private storage
    dim=384,
    model_file="onnx/model.onnx",  # can be used to load an already supported model with another optimization or quantization, e.g. onnx/model_O4.onnx
)
model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
embeddings = list(model.embed(documents))
```


### 🔱 Sparse text embeddings

* SPLADE++

```python
from fastembed import SparseTextEmbedding

model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
embeddings = list(model.embed(documents))

# [
#   SparseEmbedding(indices=[ 17, 123, 919, ... ], values=[0.71, 0.22, 0.39, ...]),
#   SparseEmbedding(indices=[ 38,  12,  91, ... ], values=[0.11, 0.22, 0.39, ...])
# ]
```

<!--
* BM42 - ([link](ToDo))

```
from fastembed import SparseTextEmbedding

model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
embeddings = list(model.embed(documents))

# [
#   SparseEmbedding(indices=[ 17, 123, 919, ... ], values=[0.71, 0.22, 0.39, ...]),
#   SparseEmbedding(indices=[ 38,  12,  91, ... ], values=[0.11, 0.22, 0.39, ...])
# ]
```
-->

### 🦥 Late interaction models (aka ColBERT)


```python
from fastembed import LateInteractionTextEmbedding

model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
embeddings = list(model.embed(documents))

# [
#   array([
#       [-0.1115,  0.0097,  0.0052,  0.0195, ...],
#       [-0.1019,  0.0635, -0.0332,  0.0522, ...],
#   ]),
#   array([
#       [-0.9019,  0.0335, -0.0032,  0.0991, ...],
#       [-0.2115,  0.8097,  0.1052,  0.0195, ...],
#   ]),  
# ]
```

### 🖼️ Image embeddings

```python
from fastembed import ImageEmbedding

images = [
    "./path/to/image1.jpg",
    "./path/to/image2.jpg",
]

model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
embeddings = list(model.embed(images))

# [
#   array([-0.1115,  0.0097,  0.0052,  0.0195, ...], dtype=float32),
#   array([-0.1019,  0.0635, -0.0332,  0.0522, ...], dtype=float32)
# ]
```

### Late interaction multimodal models (ColPali)

```python
from fastembed import LateInteractionMultimodalEmbedding

doc_images = [
    "./path/to/qdrant_pdf_doc_1_screenshot.jpg",
    "./path/to/colpali_pdf_doc_2_screenshot.jpg",
]

query = "What is Qdrant?"

model = LateInteractionMultimodalEmbedding(model_name="Qdrant/colpali-v1.3-fp16")
doc_images_embeddings = list(model.embed_image(doc_images))
# shape (2, 1030, 128)
# [array([[-0.03354, -0.0209, ..., -0.1558, -0.0768]], dtype=float16)]
query_embedding = model.embed_text(query)
# shape (1, 20, 128)
# [array([[-0.002182, 0.1476, ..., -0.02208, 0.1683]], dtype=float16)]
```

### 🔄 Rerankers
```python
from fastembed.rerank.cross_encoder import TextCrossEncoder

query = "Who is maintaining Qdrant?"
documents: list[str] = [
    "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
    "fastembed is supported by and maintained by Qdrant.",
]
encoder = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
scores = list(encoder.rerank(query, documents))

# [-11.48061752319336, 5.472434997558594]
```

Text cross encoders can also be extended with models which are not in the list of supported models.

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder 
from fastembed.common.model_description import ModelSource

TextCrossEncoder.add_custom_model(
    model="Xenova/ms-marco-MiniLM-L-4-v2",
    model_file="onnx/model.onnx",
    sources=ModelSource(hf="Xenova/ms-marco-MiniLM-L-4-v2"),
)
model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-4-v2")
scores = list(model.rerank_pairs(
    [("What is AI?", "Artificial intelligence is ..."), ("What is ML?", "Machine learning is ..."),]
))
```

## ⚡️ FastEmbed on a GPU

FastEmbed supports running on GPU devices.
It requires installation of the `fastembed-gpu` package.

```bash
pip install fastembed-gpu
```

Check our [example](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/) for detailed instructions, CUDA 12.x support and troubleshooting of the common issues.

```python
from fastembed import TextEmbedding

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    providers=["CUDAExecutionProvider"]
)
print("The model BAAI/bge-small-en-v1.5 is ready to use on a GPU.")

```

## Usage with Qdrant

Installation with Qdrant Client in Python:

```bash
pip install qdrant-client[fastembed]
```

or 

```bash
pip install qdrant-client[fastembed-gpu]
```

You might have to use quotes ```pip install 'qdrant-client[fastembed]'``` on zsh.

```python
from qdrant_client import QdrantClient, models

# Initialize the client
client = QdrantClient("localhost", port=6333) # For production
# client = QdrantClient(":memory:") # For experimentation

model_name = "sentence-transformers/all-MiniLM-L6-v2"
payload = [
    {"document": "Qdrant has Langchain integrations", "source": "Langchain-docs", },
    {"document": "Qdrant also has Llama Index integrations", "source": "LlamaIndex-docs"},
]
docs = [models.Document(text=data["document"], model=model_name) for data in payload]
ids = [42, 2]

client.create_collection(
    "demo_collection",
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), distance=models.Distance.COSINE)
)

client.upload_collection(
    collection_name="demo_collection",
    vectors=docs,
    ids=ids,
    payload=payload,
)

search_result = client.query_points(
    collection_name="demo_collection",
    query=models.Document(text="This is a query document", model=model_name)
).points
print(search_result)
```
