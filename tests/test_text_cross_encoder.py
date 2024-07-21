import unittest
from unittest.mock import patch, MagicMock
from fastembed.rerank.cross_encoder import TextCrossEncoder
import numpy as np

class TestTextCrossEncoder(unittest.TestCase):
    @patch('fastembed.rerank.cross_encoder.onnx_text_cross_encoder.OnnxCrossEncoderModel.load_onnx_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained')
    def test_rerank(self, mock_model, mock_tokenizer, mock_load_onnx_model):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_tensors.return_value = {"input_ids": np.array([[1, 2, 3], [4, 5, 6]])}
        mock_model.return_value = MagicMock()
        mock_model.return_value.run.return_value = [np.array([1.0, 0.5])]
        mock_load_onnx_model.return_value = None

        tce = TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')
        query = "What is the capital of France?"
        documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
        scores = tce.rerank(query, documents)

        self.assertEqual(len(scores), 2)
        self.assertEqual(scores, [1.0, 0.5])

    @patch('fastembed.rerank.cross_encoder.onnx_text_cross_encoder.OnnxCrossEncoderModel.load_onnx_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained')
    def test_empty_documents(self, mock_model, mock_tokenizer, mock_load_onnx_model):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_tensors.return_value = {"input_ids": np.array([])}
        mock_model.return_value = MagicMock()
        mock_model.return_value.run.return_value = [np.array([])]
        mock_load_onnx_model.return_value = None

        tce = TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')
        query = "What is the capital of France?"
        documents = []
        scores = tce.rerank(query, documents)

        self.assertEqual(len(scores), 0)

    @patch('fastembed.rerank.cross_encoder.onnx_text_cross_encoder.OnnxCrossEncoderModel.load_onnx_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained')
    def test_empty_query(self, mock_model, mock_tokenizer, mock_load_onnx_model):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_tensors.return_value = {"input_ids": np.array([[1, 2, 3], [4, 5, 6]])}
        mock_model.return_value = MagicMock()
        mock_model.return_value.run.return_value = [np.array([0.0, 0.0])]
        mock_load_onnx_model.return_value = None

        tce = TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')
        query = ""
        documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
        scores = tce.rerank(query, documents)

        self.assertEqual(len(scores), 2)
        self.assertEqual(scores, [0.0, 0.0])

    @patch('fastembed.rerank.cross_encoder.onnx_text_cross_encoder.OnnxCrossEncoderModel.load_onnx_model')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained')
    def test_long_documents(self, mock_model, mock_tokenizer, mock_load_onnx_model):
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.return_tensors.return_value = {"input_ids": np.array([[1, 2, 3], [4, 5, 6]])}
        mock_model.return_value = MagicMock()
        mock_model.return_value.run.return_value = [np.array([1.0, 0.5])]
        mock_load_onnx_model.return_value = None

        tce = TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')
        query = "What is the capital of France?"
        documents = ["Paris is the capital of France." * 100, "Berlin is the capital of Germany." * 100]
        scores = tce.rerank(query, documents)

        self.assertEqual(len(scores), 2)
        self.assertEqual(scores, [1.0, 0.5])

if __name__ == '__main__':
    unittest.main()
