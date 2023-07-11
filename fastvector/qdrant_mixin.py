from warnings import warn
from qdrant_client.models import PointStruct
from qdrant_client.http import models
from .embedding import SentenceTransformersEmbedding, OpenAIEmbedding


class QdrantClientMixin:
    def upsert_docs(
        self,
        collection_name,
        docs,
        batch_size=512,
        wait=True,
        embedding_model=None,
    ):
        if embedding_model is None:
            embedding_model = SentenceTransformersEmbedding()
        n = len(docs["documents"])

        if "ids" not in docs:
            # If not, generate sequential IDs
            docs["ids"] = list(range(1, n + 1))

        for i in range(0, n, batch_size):
            batch_docs = docs["documents"][i : i + batch_size]
            batch_metadatas = docs["metadatas"][i : i + batch_size]
            batch_ids = docs["ids"][i : i + batch_size]

            # Tokenize, embed, and index each document
            embeddings = embedding_model.encode(batch_docs)

            # Create a PointStruct for each document
            points = [
                PointStruct(id=id, vector=embedding.tolist(), payload=metadata)
                for id, embedding, metadata in zip(batch_ids, embeddings, batch_metadatas)
            ]

            # Check if collection exists
            if collection_name not in self.get_collections():
                warn(f"Collection {collection_name} not found. Creating it.")
                self.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE),
                )
            print(f"Upserting {len(points)} points")
            # Call the existing upsert method with the new PointStruct
            self.upsert(collection_name=collection_name, points=points, wait=wait)

    def query(
        self,
        collection_name,
        query_texts,
        n_results=2,
        query_filter=None,
        search_params=None,
        embedding_model=None,
        **kwargs,
    ):
        if embedding_model is None:
            embedding_model = SentenceTransformersEmbedding()

        # Encode the query text
        query_vector = embedding_model.encode(query_texts)[0]

        # Define default search parameters if not provided
        if search_params is None:
            search_params = models.SearchParams(hnsw_ef=128, exact=False)

        # Perform the search
        search_result = self.search(
            collection_name=collection_name,
            query_filter=query_filter,
            search_params=search_params,
            query_vector=query_vector.tolist(),
            limit=n_results,
            **kwargs,
        )

        return search_result
