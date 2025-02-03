from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from fastembed.sparse.bm25 import Bm25
from fastembed.rerank.cross_encoder import TextCrossEncoder


text_embedder = TextEmbedding(cache_dir="models")
late_interaction_embedder = LateInteractionTextEmbedding(model_name="", cache_dir="models")
reranker = TextCrossEncoder(model_name="", cache_dir="models")
sparse_embedder = SparseTextEmbedding(model_name="", cache_dir="models")
bm25_embedder = Bm25(model_name="", cache_dir="models")

text_embedder.list_supported_models()
text_embedder.embed(documents=[""], batch_size=1, parallel=1)
text_embedder.embed(documents="", parallel=None)
text_embedder.embed(documents="", batch_size=1, parallel=None, task_id=1)
text_embedder.query_embed(query=[""], batch_size=1, parallel=1)
text_embedder.query_embed(query="", parallel=None)
text_embedder.passage_embed(texts=[""], batch_size=1, parallel=1)
text_embedder.passage_embed(texts=[""], parallel=None)

late_interaction_embedder.list_supported_models()
late_interaction_embedder.embed(documents=[""], batch_size=1, parallel=1)
late_interaction_embedder.embed(documents="", parallel=None)
late_interaction_embedder.embed(documents="", batch_size=1, parallel=None)
late_interaction_embedder.query_embed(query=[""], batch_size=1, parallel=1)
late_interaction_embedder.query_embed(query="", parallel=None)
late_interaction_embedder.passage_embed(texts=[""], batch_size=1, parallel=1)
late_interaction_embedder.passage_embed(texts=[""], parallel=None)

reranker.list_supported_models()
reranker.rerank(query="", documents=[""], batch_size=1, parallel=1)
reranker.rerank(query="", documents=[""], parallel=None)
reranker.rerank_pairs(pairs=[("", "")], batch_size=1, parallel=1)
reranker.rerank_pairs(pairs=[("", "")], parallel=None)

sparse_embedder.list_supported_models()
sparse_embedder.embed(documents=[""], batch_size=1, parallel=1)
sparse_embedder.embed(documents="", batch_size=1, parallel=None)
sparse_embedder.embed(documents="", batch_size=1, parallel=None)
sparse_embedder.query_embed(query=[""], batch_size=1, parallel=1)
sparse_embedder.query_embed(query="", batch_size=1, parallel=None)
sparse_embedder.passage_embed(texts=[""], batch_size=1, parallel=1)
sparse_embedder.passage_embed(texts=[""], batch_size=1, parallel=None)

bm25_embedder.list_supported_models()
bm25_embedder.embed(documents=[""], batch_size=1, parallel=1)
bm25_embedder.embed(documents="", batch_size=1, parallel=None)
bm25_embedder.embed(documents="", batch_size=1, parallel=None, task_id=1)
bm25_embedder.query_embed(query=[""], batch_size=1, parallel=1)
bm25_embedder.query_embed(query="", batch_size=1, parallel=None)
bm25_embedder.raw_embed(documents=[""])
