# ⚡️ What is FastEmbed?

FastEmbed is a lightweight, fast, Python library built for embedding generation. We [support popular text models](https://qdrant.github.io/fastembed/examples/Supported_Models/). Please [open a GitHub issue](https://github.com/qdrant/fastembed/issues/new) if you want us to add a new model.

The default text embedding (`TextEmbedding`) model is Flag Embedding, the top model in the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard. It supports "query" and "passage" prefixes for the input text. Here is an example for [Retrieval Embedding Generation](https://qdrant.github.io/fastembed/examples/Retrieval_with_FastEmbed/) and how to use [FastEmbed with Qdrant](https://qdrant.github.io/fastembed/examples/Usage_With_Qdrant/).

## 📈 Why FastEmbed?

1. Light: FastEmbed is a lightweight library with few external dependencies. We don't require a GPU and don't download GBs of PyTorch dependencies, and instead use the ONNX Runtime. This makes it a great candidate for serverless runtimes like AWS Lambda. 

2. Fast: FastEmbed is designed for speed. We use the ONNX Runtime, which is faster than PyTorch. We also use data-parallelism for encoding large datasets.

3. Accurate: FastEmbed is better than OpenAI Ada-002. We also [supported](https://qdrant.github.io/fastembed/examples/Supported_Models/) an ever expanding set of models, including a few multilingual models.

## 🚀 Installation

To install the FastEmbed library, pip works:

```bash
pip install fastembed
```

## 📖 Quickstart

```python
import numpy as np
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

## Usage with Qdrant

Installation with Qdrant Client in Python:

```bash
pip install qdrant-client[fastembed]
```

You might have to use ```pip install 'qdrant-client[fastembed]'``` on zsh.

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

#### Similar Work

Ilyas M. wrote about using [FlagEmbeddings with Optimum](https://twitter.com/IlysMoutawwakil/status/1705215192425288017) over CUDA.
