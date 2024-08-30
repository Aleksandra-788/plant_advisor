from typing import List, Any
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from chromadb.utils.embedding_functions import create_langchain_embedding
import chromadb
from langchain_chroma import Chroma


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

    def create_embedding_vectorstore(self) -> None:
        """
        Embeds the data from the CSV file and stores it in the ChromaDB collection.
        """
        df = self._read_csv_file()
        self._embed_csv_data(df)

    def create_chroma_database(self) -> Chroma:
        """
        Creates a Chroma database from the embedded data.

        Returns:
            Chroma: A Chroma vector store object initialized with the embedded data.
        """
        ef = self._define_embedding_function()
        db = Chroma(client=self.client, collection_name=self.collection_name, embedding_function=ef)
        return db

    def check_collections_names(self) -> None:
        """
        Prints the names of all collections currently stored in the ChromaDB client.
        """
        collections = self.client.list_collections()
        collection_names = [collection.name for collection in collections]
        print("Collections names in client ChromaDB:")
        for name in collection_names:
            print(name)

    def collection_exists_and_filled(self) -> bool:
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
            print(f"Collection check error: {e}")
        return False

    def _read_csv_file(self) -> pd.DataFrame:
        """
        Reads the CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing the CSV data.
        """
        df = pd.read_csv(self.csv_file_path)
        return df

    def _get_column_names(self, df: pd.DataFrame) -> List[str]:
        """
        Retrieves the column names from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which to extract column names.

        Returns:
            List[str]: A list of column names in the DataFrame.
        """
        column_names = df.columns
        return column_names

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
                # column_names[4]: row[column_names[4]],
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

    def _embed_csv_data(self, df: pd.DataFrame) -> None:
        """
        Embeds the data from the DataFrame and adds it to the ChromaDB collection in batches.

        Args:
            df (pd.DataFrame): The DataFrame containing data to be embedded and stored.
        """
        column_names = self._get_column_names(df)
        batch_size = 100
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
        start_id = 1
        for i in range(num_batches):
            print(f"Batch number: {i}")
            batch = df[i * batch_size:(i + 1) * batch_size]
            self._process_batch(batch, start_id, column_names)
            start_id += batch_size

        num_documents = self.collection.count()
        print(f"Number of documents in the collection: {num_documents}")

        if num_documents == len(df):
            print("All documents have been successfully added to the collection.")
        else:
            print(f"Some documents are missing in the collection. Expected {len(df)}, but got {num_documents}.")


