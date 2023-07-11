from qdrant_client import QdrantClient

# Initialize the client
client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

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
    1,
    2,
    3,
    50,
    63,
]  # unique for each doc, if not mentioned, we'll insert this sequentially

# Use the new upsert_docs method
client.upsert_docs(
    collection_name="demo_collection",
    docs={"documents": docs, "metadatas": metadatas, "ids": ids},
    batch_size=512,  # Adjust as needed
    wait=True,  # Wait for the operation to complete
)
