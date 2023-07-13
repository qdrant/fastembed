from qdrant_client import QdrantClient

# Initialize the client
client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
metadatas = [
    {"source": "Langchain-docs"},
    {"source": "LlamaIndex-docs"},
]
ids = [42, 2]

# Use the new add method
client.add(collection_name="demo_collection", docs={"documents": docs, "metadatas": metadatas, "ids": ids})

search_result = client.query(collection_name="demo_collection", query_texts=["This is a query document"])
print(search_result)
