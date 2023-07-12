# FastVector Library

FastVector is a Python library that provides convenient methods for indexing and searching text documents using Qdrant, a high-dimensional vector indexing and search system.

## Features

- Batch document insertion with automatic embedding using SentenceTransformers. With support for OpenAI and custom embeddings.
- Efficient batch searching with support for filtering by metadata.
- Automatic generation of unique IDs for documents.
- Convenient alias methods for adding documents and performing queries.

## Installation

To install the FastVector library, use pip:

```bash
pip install fastvector
```

## Usage

Here's a simple usage example, which works as is:

```python
from qdrant_client import QdrantClient

# Initialize the client
client = QrantClient(":memory:")  # or QdrantClient(path="path/to/db")

# Prepare your documents, metadata, and IDs
docs = [
    "Qdrant has Langchain integrations",
    "Qdrant also has Llama Index integrations",
    # ...more documents...
]
metadatas = [
    {"source": "notion"},
    {"source": "google-docs"},
    # ...more metadata...
]
ids = [
    42,
    2,
]  # unique for each doc, if not mentioned, we'll generate random IDs, can lead to duplicates

# Use the new add method
client.add(
    collection_name="demo_collection",
    docs={"documents": docs, "metadatas": metadatas, "ids": ids},
    batch_size=512,  # Adjust as needed
    wait=True,  # Wait for the operation to complete
)

search_result = client.query(
    collection_name="demo_collection",
    query_texts=["This is a query document"],
    n_results=2,
)
print(search_result)
```
