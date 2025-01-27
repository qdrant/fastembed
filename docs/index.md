# ⚡️ What is FastEmbed?

FastEmbed is a lightweight, fast, Python library built for embedding generation. We [support popular text models](https://qdrant.github.io/fastembed/examples/Supported_Models/). Please [open a Github issue](https://github.com/qdrant/fastembed/issues/new) if you want us to add a new model.

1. Light & Fast
    - Quantized model weights
    - ONNX Runtime for inference

2. Accuracy/Recall
    - Better than OpenAI Ada-002
    - Default is Flag Embedding, which has shown good results on the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard
    - List of [supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/) - including multilingual models

Here is an example for [Retrieval Embedding Generation](https://qdrant.github.io/fastembed/examples/Retrieval%20with%20FastEmbed/) and how to use [FastEmbed with Qdrant](https://qdrant.github.io/fastembed/examples/Usage_With_Qdrant/).

## 🚀 Installation

To install the FastEmbed library, pip works:

```bash
pip install fastembed
```

## 📖 Usage

```python
import numpy as np
from numpy.typing import NDArray

from fastembed import TextEmbedding

documents: list[str] = [
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    "fastembed is supported by and maintained by Qdrant."
]
embedding_model = TextEmbedding()
embeddings: list[NDArray[np.float32]] = embedding_model.embed(documents)
```

## Usage with Qdrant

Installation with Qdrant Client in Python:

```bash
pip install qdrant-client[fastembed]
```

Might have to use ```pip install 'qdrant-client[fastembed]'``` on zsh.

```python
from qdrant_client import QdrantClient

# Initialize the client
client = QdrantClient(":memory:")  # Using an in-process Qdrant

# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
metadata = [
    {"source": "Langchain-docs"},
    {"source": "Llama-index-docs"},
]
ids = [42, 2]

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
