from warnings import warn
from qdrant_client.models import PointStruct
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


class QdrantClientMixin:
    def upsert_docs(self, collection_name, docs, batch_size=512, wait=True):
        # Initialize the SentenceTransformer model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        n = len(docs["documents"])
        for i in range(0, n, batch_size):
            batch_docs = docs["documents"][i : i + batch_size]
            batch_metadatas = docs["metadatas"][i : i + batch_size]
            batch_ids = docs["ids"][i : i + batch_size]

            # Tokenize, embed, and index each document
            embeddings = model.encode(batch_docs)

            # Create a PointStruct for each document
            points = [
                PointStruct(id=id, vector=embedding.tolist(), payload=metadata)
                for id, embedding, metadata in zip(
                    batch_ids, embeddings, batch_metadatas
                )
            ]

            # Check if collection exists
            if collection_name not in self.get_collections():
                warn(f"Collection {collection_name} not found. Creating it.")
                self.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=len(embeddings[0]), distance=models.Distance.COSINE
                    ),
                )
            print(f"Upserting {len(points)} points")
            # Call the existing upsert method with the new PointStruct
            self.upsert(collection_name=collection_name, points=points, wait=wait)
