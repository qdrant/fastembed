# FastEmbed Library

FastEmbed is a Python library that provides convenient methods for indexing and searching text documents

## Installation

To install the FastEmbed library, we recommend using Poetry, alternatively -- pip works: 

```bash
pip install fastembed
```

## Usage

```python
from fastembed.embedding import DefaultEmbedding

documents: List[str] = [
    "Hello, World!",
    "This is an example document.",
    "fastembed is supported by and maintained by Qdrant." * 128,
]
# Initialize the DefaultEmbedding class with the desired parameters
# model_name="BAAI/bge-small-en"
embedding_model = DeafultEmbedding() 
embeddings: List[np.ndarray] = list(embedding_model.encode(documents))
```
