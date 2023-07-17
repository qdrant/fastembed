# FastEmbed Library

FastEmbed is a Python library that provides convenient methods for indexing and searching text documents using Qdrant, a high-dimensional vector indexing and search system.

## Features

- Batch document insertion with automatic embedding using SentenceTransformers. With support for OpenAI and custom embeddings.
- Efficient batch searching with support for filtering by metadata.
- Automatic generation of unique IDs for documents.
- Convenient alias methods for adding documents and performing queries.

## Installation

To install the FastEmbed library, we install Qdrant client as well with pip:

```bash
pip install fastembed qdrant-client
```

## Usage

Here's a simple usage example, which works as is:

```python
from qdrant_client import QdrantClient

# Initialize the client
client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
metadatas = [
    {"source": "Langchain-docs"},
    {"source": "Linkedin-docs"},
]
ids = [42, 2]

# Use the new add method
client.add(collection_name="demo_collection", docs={"documents": docs, "metadatas": metadatas, "ids": ids})

search_result = client.query(collection_name="demo_collection", query_texts=["This is a query document"])
print(search_result)
```
