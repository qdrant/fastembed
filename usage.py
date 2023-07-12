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
    72,
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
