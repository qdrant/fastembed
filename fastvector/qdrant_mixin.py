from warnings import warn
from qdrant_client.models import PointStruct
from qdrant_client.http import models


class QdrantClientMixin:
    def upsert_docs(
        self,
        collection_name,
        docs,
        batch_size=512,
        wait=True,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Initialize the SentenceTransformer model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Please install the sentence-transformers package to use this method."
            )
        model = SentenceTransformer(model_name)

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

    def query(
        self,
        collection_name,
        query_texts,
        n_results=2,
        query_filter=None,
        search_params=None,
        **kwargs,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Please install the sentence-transformers package to use this method."
            )
        # Initialize the SentenceTransformer model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Encode the query text
        query_vector = model.encode(query_texts)[0]

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
