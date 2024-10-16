from typing import List, Any
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from chromadb.utils.embedding_functions import create_langchain_embedding
import chromadb
from langchain_chroma import Chroma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDatabaseManager:
    """
    Manages a Chroma database by embedding data from a CSV file using a HuggingFace model.

    Attributes:
        csv_file_path (str): Path to the CSV file containing plant data.
        model_name (str): Name of the HuggingFace model used for creating embeddings.
        collection_name (str): Name of the collection in the Chroma database.
        client (chromadb.PersistentClient): Persistent ChromaDB client.
        collection (chromadb.Collection): ChromaDB collection for storing embedded data.
    """

    def __init__(self,
                 csv_file_path: str = "data/plants_data.csv",
                 model_name: str = "sentence-transformers/gtr-t5-base",
                 collection_name: str = "plant_advisor_collection"):
        """
        Initializes the ChromaDatabaseManager with paths and model names.

        Args:
            csv_file_path (str): Path to the CSV file. Defaults to "../data/plants_data.csv".
            model_name (str): HuggingFace model name for embeddings. Defaults to "sentence-transformers/gtr-t5-base".
            collection_name (str): Name of the ChromaDB collection. Defaults to "plant_eng".
        """
        self.csv_file_path = csv_file_path
        self.model_name = model_name
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="chroma/")
        self.collection = self._create_collection_chromadb()
        self._check_and_create_chroma_collection()
        self.database = self._create_chroma_database()

    def _check_and_create_chroma_collection(self) -> None:
        """
           Checks if the data has been embedded and the collection has been created in ChromaDB.
           If the collection does not exist or is not filled, it creates the embedding vectorstore.
        """
        logger.info("Checking if database collection exists and is filled...")
        if not self._collection_exists_and_filled():
            logger.info("Creating embedding vectorstore...")
            self._embed_data()
        else:
            logger.info("Collection already exists and is filled with data.")

    def _embed_data(self) -> None:
        """
        Embeds the data from the CSV file and stores it in the ChromaDB collection.
        """
        df = self._read_csv_file()
        column_names = df.columns
        batch_size = 100
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
        for i in range(num_batches):
            logger.info(f"Batch number: {i}")
            logger.info(f"Left {num_batches-i} batches to finish.")
            batch = df[i * batch_size:(i + 1) * batch_size]
            self._process_batch(batch, i * batch_size + 1, column_names)
        num_documents = self.collection.count()
        logger.info(f"Number of documents in the collection: {num_documents}")
        if num_documents == len(df):
            logger.info("All documents have been successfully added to the collection.")
        else:
            logger.info(f"Some documents are missing in the collection. Expected {len(df)}, but got {num_documents}.")

    def _create_chroma_database(self) -> Chroma:
        """
        Creates a Chroma database from the embedded data.

        Returns:
            Chroma: A Chroma vector store object initialized with the embedded data.
        """
        ef = self._define_embedding_function()
        db = Chroma(client=self.client, collection_name=self.collection_name, embedding_function=ef)
        return db

    def _collection_exists_and_filled(self) -> bool:
        """
        Checks if the ChromaDB collection exists and is filled with documents.

        Returns:
            bool: True if the collection exists and has documents, False otherwise.
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            if collection.count() > 0:
                return True
        except Exception as e:
            logger.info(f"Collection check error: {e}")
        return False

    def _read_csv_file(self) -> pd.DataFrame:
        """
        Reads the CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing the CSV data.
        """
        df = pd.read_csv(self.csv_file_path)
        return df

    def _define_embedding_function(self) -> Any:
        """
        Defines the embedding function using a HuggingFace model.

        Returns:
            Any: The embedding function compatible with ChromaDB.
        """
        langchain_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        ef = create_langchain_embedding(langchain_embeddings)
        return ef

    def _create_collection_chromadb(self) -> chromadb.Collection:
        """
        Creates or retrieves a ChromaDB collection.

        Returns:
            chromadb.Collection: The ChromaDB collection object.
        """
        ef = self._define_embedding_function()
        collection = self.client.get_or_create_collection(name=self.collection_name, embedding_function=ef)
        return collection

    def _process_batch(self, batch: pd.DataFrame, start_id: int, column_names: List[str]) -> None:
        """
        Processes a batch of data and adds it to the ChromaDB collection.

        Args:
            batch (pd.DataFrame): A batch of data from the CSV file.
            start_id (int): The starting ID for the documents in this batch.
            column_names (List[str]): The names of the columns in the DataFrame.
        """
        documents = []
        metadatas = []
        ids = []
        for _, row in batch.iterrows():
            documents.append(row[column_names[6]])
            metadatas.append({
                column_names[0]: row[column_names[0]],
                column_names[1]: row[column_names[1]],
                column_names[2]: row[column_names[2]],
                column_names[3]: row[column_names[3]],
                column_names[5]: row[column_names[5]],
                column_names[7]: row[column_names[7]]
            })
            ids.append(str(start_id))
            start_id += 1
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
