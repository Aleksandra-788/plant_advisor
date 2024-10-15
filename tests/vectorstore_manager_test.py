import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.chroma_database_manager import ChromaDatabaseManager


class TestChromaDatabaseManager(unittest.TestCase):

    @patch('src.vectorstore_manager.chromadb.PersistentClient')
    def setUp(self, mock_persistent_client):
        self.mock_client = mock_persistent_client.return_value
        self.manager = ChromaDatabaseManager(
            csv_file_path="test_data/plants_data.csv",
            model_name="sentence-transformers/gtr-t5-base",
            collection_name="test_collection"
        )
        self.mock_collection = MagicMock()
        self.manager.collection = self.mock_collection

    @patch('src.vectorstore_manager.pd.read_csv')
    def test_read_csv_file(self, mock_read_csv):
        df_mock = pd.DataFrame({"column1": [1, 2, 3]})
        mock_read_csv.return_value = df_mock

        result = self.manager._read_csv_file()
        mock_read_csv.assert_called_once_with("test_data/plants_data.csv")
        self.assertEqual(result.shape, (3, 1))

    def test_get_column_names(self):
        df_mock = pd.DataFrame({
            "name": ["Plant 1", "Plant 2"],
            "type": ["Type A", "Type B"]
        })
        columns = self.manager._get_column_names(df_mock)
        self.assertEqual(list(columns), ["name", "type"])

    @patch('src.vectorstore_manager.HuggingFaceEmbeddings')
    @patch('src.vectorstore_manager.create_langchain_embedding')
    def test_define_embedding_function(self, mock_create_embedding, mock_hf_embeddings):
        mock_embeddings_instance = mock_hf_embeddings.return_value
        mock_embedding_function = mock_create_embedding.return_value

        result = self.manager._define_embedding_function()

        mock_hf_embeddings.assert_called_once_with(model_name="sentence-transformers/gtr-t5-base")
        mock_create_embedding.assert_called_once_with(mock_embeddings_instance)
        self.assertEqual(result, mock_embedding_function)

    @patch('src.vectorstore_manager.HuggingFaceEmbeddings')
    @patch('src.vectorstore_manager.create_langchain_embedding')
    def test_create_collection_chromadb(self, mock_create_embedding, mock_hf_embeddings):
        mock_embeddings_instance = mock_hf_embeddings.return_value
        mock_embedding_function = mock_create_embedding.return_value
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        result = self.manager._create_collection_chromadb()

        self.assertEqual(result, self.mock_collection)

    def test_check_collections_names(self):
        mock_collection1 = MagicMock()
        mock_collection1.name = "collection_1"
        mock_collection2 = MagicMock()
        mock_collection2.name = "collection_2"
        self.mock_client.list_collections.return_value = [mock_collection1, mock_collection2]

        with patch('builtins.print') as mocked_print:
            self.manager.check_collections_names()

        self.mock_client.list_collections.assert_called_once()
        mocked_print.assert_any_call("Collections names in client ChromaDB:")
        mocked_print.assert_any_call("collection_1")
        mocked_print.assert_any_call("collection_2")

    def test_collection_exists_and_filled(self):
        self.mock_collection.count.return_value = 10
        self.mock_client.get_collection.return_value = self.mock_collection
        result = self.manager.collection_exists_and_filled()
        self.mock_client.get_collection.assert_called_once_with(name="test_collection")
        self.assertTrue(result)

    def test_collection_exists_and_empty(self):
        self.mock_collection.count.return_value = 0
        self.mock_client.get_collection.return_value = self.mock_collection
        result = self.manager.collection_exists_and_filled()
        self.mock_client.get_collection.assert_called_once_with(name="test_collection")
        self.assertFalse(result)

    def test_collection_exists_and_filled_exception(self):
        self.mock_client.get_collection.side_effect = Exception("Collection not found")
        result = self.manager.collection_exists_and_filled()
        self.assertFalse(result)

    def test_process_batch(self):

        df_mock = pd.DataFrame({
            "col0": ["Plant A", "Plant B"],
            "col1": ["Type 1", "Type 2"],
            "col2": ["Description A", "Description B"],
            "col3": ["Metadata A", "Metadata B"],
            "col4": ["Metadata C", "Metadata D"],
            "col5": ["Metadata E", "Metadata F"],
            "col6": ["Doc A", "Doc B"],
            "col7": ["Metadata E", "Metadata F"]
        })

        column_names = df_mock.columns
        self.manager._process_batch(df_mock, start_id=1, column_names=column_names)

        self.mock_collection.add.assert_called_once_with(
            documents=["Doc A", "Doc B"],
            metadatas=[
                {'col0': 'Plant A', 'col1': 'Type 1', 'col2': 'Description A', 'col3': 'Metadata A',
                 'col5': 'Metadata E', 'col7': 'Metadata E'},
                {'col0': 'Plant B', 'col1': 'Type 2', 'col2': 'Description B', 'col3': 'Metadata B',
                 'col5': 'Metadata F', 'col7': 'Metadata F'}
            ],
            ids=["1", "2"]
        )

    @patch.object(ChromaDatabaseManager, '_embed_csv_data')
    @patch.object(ChromaDatabaseManager, '_read_csv_file')
    def test_create_embedding_vectorstore(self, mock_read_csv, mock_embed_csv):
        df_mock = pd.DataFrame({"col1": ["data1", "data2"]})
        mock_read_csv.return_value = df_mock
        self.manager.embed_data()
        mock_read_csv.assert_called_once()
        mock_embed_csv.assert_called_once_with(df_mock)

    @patch('src.vectorstore_manager.Chroma')
    @patch.object(ChromaDatabaseManager, '_define_embedding_function')
    def test_create_chroma_database(self, mock_define_embedding_function, mock_chroma):
        mock_embedding_function = MagicMock()
        mock_define_embedding_function.return_value = mock_embedding_function
        db = self.manager.create_chroma_database()
        self.assertEqual(db, mock_chroma.return_value)

    @patch.object(ChromaDatabaseManager, '_process_batch')
    @patch.object(ChromaDatabaseManager, '_get_column_names')
    def test_embed_csv_data(self, mock_get_column_names, mock_process_batch):
        mock_get_column_names.return_value = ["col1", "col2", "col3", "col4", "col5", "col6", "col7"]
        df_mock = pd.DataFrame({
            "col1": ["data1", "data2", "data3"],
            "col2": ["data4", "data5", "data6"]
        })
        self.manager._embed_csv_data(df_mock)
        mock_get_column_names.assert_called_once_with(df_mock)
        mock_process_batch.assert_called()

