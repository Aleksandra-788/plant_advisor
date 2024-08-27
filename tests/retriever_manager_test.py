import pytest
from unittest.mock import MagicMock, patch
from src.retriever_manager import RAGManager
from src.prompt_manager import PromptManager
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_community.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
import sys
sys.path.append(".")


@pytest.fixture
@patch("src.retriever_manager.ChatOpenAI")
def rag_manager(mock_chat_openai, mock_vectorstore):
    mock_chat_openai.return_value = MagicMock()
    return RAGManager(prompt_manager=PromptManager(), vectorstore=mock_vectorstore)



@pytest.fixture
def mock_vectorstore():
    return MagicMock(VectorStore)


def test_create_retriever(rag_manager, mock_vectorstore):
    plant_groups = ["perennial", "deciduous"]
    retriever = rag_manager.create_retriever(plant_groups)

    # assert isinstance(retriever, BaseRetriever)
    filter_dict = {'plant group': {"$in": plant_groups}}
    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": rag_manager.k, 'filter': filter_dict})


def test_create_rag_chain(rag_manager):
    plant_groups = ["perennial", "deciduous"]
    mock_retriever = MagicMock(BaseRetriever)
    rag_manager.create_retriever = MagicMock(return_value=mock_retriever)

    rag_chain = rag_manager._create_rag_chain(plant_groups)
    # assert isinstance(rag_chain, Runnable)
    rag_manager.create_retriever.assert_called_once_with(plant_groups)


@patch("src.retriever_manager.RAGManager._create_rag_chain")
def test_get_response(mock_create_rag_chain, rag_manager):
    plant_groups = ["perennial", "deciduous"]
    plant_description = "A plant with broad leaves."
    mock_chain = MagicMock(Runnable)
    mock_create_rag_chain.return_value = mock_chain
    mock_chain.invoke.return_value = "Generated response"

    response = rag_manager.get_response(plant_groups, plant_description)

    assert response == "Generated response"
    mock_create_rag_chain.assert_called_once_with(plant_groups=plant_groups)
    mock_chain.invoke.assert_called_once_with(input=plant_description)

