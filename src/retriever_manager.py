from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from src.prompt_manager import PromptManager
from typing import List, Optional
from langchain_community.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever


class RAGManager:
    def __init__(self, prompt_manager: PromptManager, vectorstore: VectorStore, llm: Optional[ChatOpenAI] = None,
                 k: int = 5):
        """
                Initializes the RAGManager with a vector store and a language model.

                Args:
                    vectorstore (VectorStore): The vector store used for retrieving documents.
                    llm (Optional[ChatOpenAI]): The language model used for generating responses.
                                                Defaults to a GPT-4 model if not provided.
                    k (int): The number of documents to retrieve in the RAG process. Defaults to 5.
                """
        self.rag_prompt = prompt_manager.create_prompt(file_name="rag_template")
        self.vectorstore = vectorstore
        self.llm = llm if llm else ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=256)
        self.k = k

    def create_retriever(self, plant_groups: List[str]) -> BaseRetriever:
        """
                Creates a retriever that filters documents based on the provided plant groups.

                Args:
                    plant_groups (List[str]): A list of plant groups to filter the search.

                Returns:
                    Retriever: An object that retrieves relevant documents based on the plant groups.
                """
        filter_dict = {'plant group': {"$in": plant_groups}}
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k, 'filter': filter_dict})
        return retriever

    def _create_rag_chain(self, plant_groups: List[str], ) -> Runnable:
        """
                Creates a RAG (Retrieval-Augmented Generation) chain for generating responses.

                Args:
                    plant_groups (List[str]): A list of plant groups used to filter the retriever.

                Returns:
                    Chain (Runnable): A chain object that links the retriever, prompt, and language model
                    to generate a response.
                """
        retriever = self.create_retriever(plant_groups)
        rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()}
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def get_response(self, plant_groups: List[str], plant_description: str) -> str:
        """
                Generates a response based on the provided plant groups and description.

                Args:
                    plant_groups (List[str]): A list of plant groups used to filter the retriever.
                    plant_description (str): The description of the plant to be used as input for
                    the response generation.

                Returns:
                    str: The generated response based on the RAG chain.
                """
        rag_chain = self._create_rag_chain(plant_groups=plant_groups)
        response = rag_chain.invoke(input=plant_description)
        return response
