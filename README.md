# ⚡️ What is FastEmbed?

FastEmbed is a lightweight, fast, Python library built for embedding generation. We [support popular text models](https://qdrant.github.io/fastembed/examples/Supported_Models/). Please [open a Github issue](https://github.com/qdrant/fastembed/issues/new) if you want us to add a new model.

The default embedding supports "query" and "passage" prefixes for the input text. The default model is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard. Here is an example for [Retrieval Embedding Generation](https://qdrant.github.io/fastembed/examples/Retrieval_with_FastEmbed/) and how to use [FastEmbed with Qdrant](https://qdrant.github.io/fastembed/examples/Usage_With_Qdrant/).

1. Light & Fast
    - Quantized model weights
    - ONNX Runtime, no PyTorch dependency
    - CPU-first design
    - Data-parallelism for encoding of large datasets

2. Accuracy/Recall
    - Better than OpenAI Ada-002
    - Default is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard
    - List of [supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/) - including multilingual models

## 🚀 Installation

To install the FastEmbed library, pip works: 

```bash
pip install fastembed
```

## 📖 Usage

```python
from fastembed.embedding import FlagEmbedding as Embedding
from typing import List
import numpy as np

documents: List[str] = [
    "passage: Hello, World!",
    "query: Hello, World!", # these are two different embedding
    "passage: This is an example passage.",
    "fastembed is supported by and maintained by Qdrant." # You can leave out the prefix but it's recommended
]
embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512) 
embeddings: List[np.ndarray] = list(embedding_model.embed(documents)) # Note the list() call - this is a generator 
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
client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
metadata = [
    {"source": "Langchain-docs"},
    {"source": "Linkedin-docs"},
]
ids = [42, 2]

# Use the new add method
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

#### Similar Work

Ilyas M. wrote about using [FlagEmbeddings with Optimum](https://twitter.com/IlysMoutawwakil/status/1705215192425288017) over CUDA.
