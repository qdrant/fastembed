supported_flag_models = [
    {
        "model": "BAAI/bge-base-en",
        "dim": 768,
        "description": "Base English model",
        "size_in_GB": 0.5,
        "sources": {
            "gcp": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en.tar.gz",
        },
    },
    {
        "model": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "description": "Base English model, v1.5",
        "size_in_GB": 0.44,
        "sources": {
            "gcp": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en-v1.5.tar.gz",
            "hf": "qdrant/bge-base-en-v1.5-onnx-q",
        }
    },
    {
        "model": "BAAI/bge-large-en-v1.5-quantized",
        "dim": 1024,
        "description": "Large English model, v1.5",
        "size_in_GB": 1.34,
        "sources": {
            "hf": "qdrant/bge-large-en-v1.5-onnx-q",
        }
    },
    {
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "description": "Large English model, v1.5",
        "size_in_GB": 1.34,
        "sources": {
            "hf": "qdrant/bge-large-en-v1.5-onnx",
        }
    },
    {
        "model": "BAAI/bge-small-en",
        "dim": 384,
        "description": "Fast English model",
        "size_in_GB": 0.2,
        "sources": {
            "gcp": "https://storage.googleapis.com/qdrant-fastembed/BAAI-bge-small-en.tar.gz",
        }
    },
    # {
    #     "model": "BAAI/bge-small-en",
    #     "dim": 384,
    #     "description": "Fast English model",
    #     "size_in_GB": 0.2,
    #     "hf_sources": [],
    #     "compressed_url_sources": [
    #         "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en.tar.gz",
    #         "https://storage.googleapis.com/qdrant-fastembed/BAAI-bge-small-en.tar.gz"
    #     ]
    # },
    {
        "model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "description": "Fast and Default English model",
        "size_in_GB": 0.13,
        "sources": {
            "gcp": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
            "hf": "qdrant/bge-small-en-v1.5-onnx-q",
        }
    },
    {
        "model": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "description": "Fast and recommended Chinese model",
        "size_in_GB": 0.1,
        "sources": {
            "gcp": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-zh-v1.5.tar.gz",
        }
    },
    {  # todo: it is not a flag embedding
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Sentence Transformer model, MiniLM-L6-v2",
        "size_in_GB": 0.09,
        "sources": {
            "gcp": "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
            "hf": "qdrant/all-MiniLM-L6-v2-onnx",
        }
    },
    # {
    #     "model": "sentence-transformers/all-MiniLM-L6-v2",
    #     "dim": 384,
    #     "description": "Sentence Transformer model, MiniLM-L6-v2",
    #     "size_in_GB": 0.09,
    #     "hf_sources": [
    #         "qdrant/all-MiniLM-L6-v2-onnx"
    #     ],
    #     "compressed_url_sources": [
    #         "https://storage.googleapis.com/qdrant-fastembed/fast-all-MiniLM-L6-v2.tar.gz",
    #         "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"
    #     ]
    # }
]

supported_multilingual_e5_models = [
    {
        "model": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "description": "Multilingual model, e5-large. Recommend using this model for non-English languages",
        "size_in_GB": 2.24,
        "sources": {
            "gcp": "https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
            "hf": "qdrant/multilingual-e5-large-onnx",
        }
    }
]

supported_jina_models = [
    {
        "model": "jinaai/jina-embeddings-v2-base-en",
        "dim": 768,
        "description": "English embedding model supporting 8192 sequence length",
        "size_in_GB": 0.55,
        "sources": {
            "hf": "xenova/jina-embeddings-v2-base-en"
        }
    },
    {
        "model": "jinaai/jina-embeddings-v2-small-en",
        "dim": 512,
        "description": "English embedding model supporting 8192 sequence length",
        "size_in_GB": 0.13,
        "sources": {"hf": "xenova/jina-embeddings-v2-small-en"}
    }
]
