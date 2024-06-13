# ⚡️ What is FastEmbed?

FastEmbed is a lightweight, fast, Python library built for embedding generation. We [support popular text models](https://qdrant.github.io/fastembed/examples/Supported_Models/). Please [open a GitHub issue](https://github.com/qdrant/fastembed/issues/new) if you want us to add a new model.

The default text embedding (`TextEmbedding`) model is Flag Embedding, presented in the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard. It supports "query" and "passage" prefixes for the input text. Here is an example for [Retrieval Embedding Generation](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/) and how to use [FastEmbed with Qdrant](https://qdrant.github.io/fastembed/qdrant/Usage_With_Qdrant/).

## 📈 Why FastEmbed?

1. Light: FastEmbed is a lightweight library with few external dependencies. We don't require a GPU and don't download GBs of PyTorch dependencies, and instead use the ONNX Runtime. This makes it a great candidate for serverless runtimes like AWS Lambda. 

2. Fast: FastEmbed is designed for speed. We use the ONNX Runtime, which is faster than PyTorch. We also use data-parallelism for encoding large datasets.

3. Accurate: FastEmbed is better than OpenAI Ada-002. We also [supported](https://qdrant.github.io/fastembed/examples/Supported_Models/) an ever expanding set of models, including a few multilingual models.

## 🚀 Installation

To install the FastEmbed library, pip works best. You can install it with or without GPU support:

```bash
pip install fastembed

# or with GPU support

pip install fastembed-gpu
```

## 📖 Quickstart

```python
from fastembed import TextEmbedding
from typing import List

# Example list of documents
documents: List[str] = [
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
embeddings = list(embedding_model.embed(documents))

# [
#   array([-0.1115,  0.0097,  0.0052,  0.0195, ...], dtype=float32),
#   array([-0.1019,  0.0635, -0.0332,  0.0522, ...], dtype=float32)
# ]

```



### 🔱 Sparse text embeddings

* SPLADE++

```python
from fastembed import SparseTextEmbedding

model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
embeddings = list(embedding_model.embed(documents))

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
embeddings = list(embedding_model.embed(documents))

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
embeddings = list(embedding_model.embed(documents))

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
embeddings = list(embedding_model.embed(images))

# [
#   array([-0.1115,  0.0097,  0.0052,  0.0195, ...], dtype=float32),
#   array([-0.1019,  0.0635, -0.0332,  0.0522, ...], dtype=float32)
# ]
```


## ⚡️ FastEmbed on a GPU

FastEmbed supports running on GPU devices.
It requires installation of the `fastembed-gpu` package.

```bash
pip install fastembed-gpu
```

Check our [example](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/) for the detailed instructions and CUDA 12.x support.

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
from qdrant_client import QdrantClient

# Initialize the client
client = QdrantClient("localhost", port=6333) # For production
# client = QdrantClient(":memory:") # For small experiments

# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
metadata = [
    {"source": "Langchain-docs"},
    {"source": "Llama-index-docs"},
]
ids = [42, 2]

# If you want to change the model:
# client.set_model("sentence-transformers/all-MiniLM-L6-v2")
# List of supported models: https://qdrant.github.io/fastembed/examples/Supported_Models

# Use the new add() instead of upsert()
# This internally calls embed() of the configured embedding model
client.add(
    collection_name="demo_collection",
    documents=docs,
    metadata=metadata,
    ids=ids
)

search_result = client.query(
    collection_name="demo_collection",
    query_text="This is a query document"
)
print(search_result)
```