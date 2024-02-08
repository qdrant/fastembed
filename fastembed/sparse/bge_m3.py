"""
Reference implementation of BGE_M3 model.
https://github.com/FlagOpen/FlagEmbedding/blob/e23ff5e213350cbd4bb50883f6dbbecf6c267965/FlagEmbedding/BGE_M3/modeling.py#L340
"""
from typing import Dict

import numpy as np


class BGE_M3:

    def dense_embedding_np(hidden_state, mask, sentence_pooling_method):
        if sentence_pooling_method == 'cls':
            return hidden_state[:, 0, :]
        elif sentence_pooling_method == 'mean':
            masked_hidden_state = hidden_state * mask[:, :, None]
            sum_embeddings = masked_hidden_state.sum(axis=1)
            token_counts = mask.sum(axis=1, keepdims=True)
            token_counts = np.where(token_counts == 0, 1, token_counts)
            mean_embeddings = sum_embeddings / token_counts
            return mean_embeddings


    def forward(self,
                text_input: Dict[str, Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = False,
                return_colbert: bool = False,
                return_sparse_embedding: bool = False):
        assert return_dense or return_sparse or return_colbert, 'Must choose one or more from `return_colbert`, `return_sparse`, `return_dense` to set `True`!'

        last_hidden_state = self.model(**text_input, return_dict=True).last_hidden_state

        output = {}
        if return_dense:
            dense_vecs = self.dense_embedding(last_hidden_state, text_input['attention_mask'])
            output['dense_vecs'] = dense_vecs
        if return_sparse:
            sparse_vecs = self.sparse_embedding(last_hidden_state, text_input['input_ids'],
                                                return_embedding=return_sparse_embedding)
            output['sparse_vecs'] = sparse_vecs
        if return_colbert:
            colbert_vecs = self.colbert_embedding(last_hidden_state, text_input['attention_mask'])
            output['colbert_vecs'] = colbert_vecs

        if self.normlized:
            if 'dense_vecs' in output:
                output['dense_vecs'] = torch.nn.functional.normalize(output['dense_vecs'], dim=-1)
            if 'colbert_vecs' in output:
                output['colbert_vecs'] = torch.nn.functional.normalize(output['colbert_vecs'], dim=-1)

        return output