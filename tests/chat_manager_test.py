import pytest
from src.chat_manager import ChatManager
from unittest.mock import Mock, patch, MagicMock
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import sys
sys.path.append(".")


@pytest.fixture
@patch('src.chat_manager.ChatOpenAI')
def chat_manager(mock_chat_openai):
    mock_chat_openai.return_value = MagicMock()
    mock_prompt_manager = Mock()
    mock_prompt_manager.create_prompt.return_value = MagicMock()
    return ChatManager(prompt_manager=mock_prompt_manager)


class TestChatManager:

    def test_init(self, chat_manager):
        assert isinstance(chat_manager, ChatManager)
        assert chat_manager.session_id == "s01"
        assert chat_manager.store == {}
        assert chat_manager.chat_chain_with_message_history is None

    def test_get_session_history_no_existing_history(self, chat_manager):
        history = chat_manager.get_session_history()
        assert isinstance(history, ChatMessageHistory)
        assert chat_manager.store.get(chat_manager.session_id) is not None

    def test_get_session_history_existing_history(self, chat_manager):
        mock_history = MagicMock()
        chat_manager.store[chat_manager.session_id] = mock_history
        history = chat_manager.get_session_history()
        assert history == mock_history

    def test_create_chat_chain(self, chat_manager):
        chat_chain = chat_manager.create_chat_chain()
        assert chat_chain is not None
        assert isinstance(chat_chain, RunnableWithMessageHistory)

    def test_get_response(self, chat_manager):
        chat_manager.chat_chain_with_message_history = MagicMock()
        chat_manager.chat_chain_with_message_history.invoke.return_value = "Test response"
        response = chat_manager.get_response("Hello")
        assert response == "Test response"
        chat_manager.chat_chain_with_message_history.invoke.assert_called_once_with(
            {"input": "Hello"},
            config={"configurable": {"session_id": chat_manager.session_id}}
        )

    def test_get_response_with_empty_input(self, chat_manager):
        chat_manager.chat_chain_with_message_history = MagicMock()
        chat_manager.chat_chain_with_message_history.invoke.return_value = "Empty input response"
        response = chat_manager.get_response("")
        assert response == "Empty input response"
        chat_manager.chat_chain_with_message_history.invoke.assert_called_once_with(
            {"input": ""},
            config={"configurable": {"session_id": chat_manager.session_id}}
        )

    def test_get_response_creates_chat_chain(self, chat_manager):
        chat_manager.chat_chain_with_message_history = None
        chat_manager.create_chat_chain = MagicMock()
        chat_manager.create_chat_chain.return_value = MagicMock(invoke=MagicMock(return_value="New chain response"))
        response = chat_manager.get_response("Hello")
        chat_manager.create_chat_chain.assert_called_once()
        assert response == "New chain response"

    def test_clear_session_history(self, chat_manager):
        mock_history = MagicMock()
        chat_manager.store[chat_manager.session_id] = mock_history
        chat_manager.clear_session_history()
        mock_history.clear.assert_called_once()
