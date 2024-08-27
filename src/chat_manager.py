from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.prompt_manager import PromptManager


class ChatManager:
    def __init__(self, prompt_manager: PromptManager, openai_llm_model_name: str = "gpt-4o-mini"):
        """
        Initializes the ChatManager with a specified OpenAI LLM model.

        Args:
            openai_llm_model_name (str): The name of the OpenAI language model to use. Defaults to "gpt-4o-mini".
        """
        self.llm = ChatOpenAI(model_name=openai_llm_model_name, temperature=0, max_tokens=256)
        self.chat_prompt = prompt_manager.create_prompt(file_name='conversational_template')
        self.store: Dict[str, ChatMessageHistory] = {}
        self.session_id: str = "s01"
        self.chat_chain_with_message_history: Optional[RunnableWithMessageHistory] = None

    def get_session_history(self) -> ChatMessageHistory:
        """
        Retrieves the chat message history for the current session. If no history exists, it creates a new one.

        Returns:
            ChatMessageHistory: The chat message history object for the current session.
        """
        if self.session_id not in self.store:
            self.store[self.session_id] = ChatMessageHistory()
        return self.store[self.session_id]

    def create_chat_chain(self) -> RunnableWithMessageHistory:
        """
        Creates a chat chain that includes the prompt, LLM, and message history.

        Returns:
            RunnableWithMessageHistory: A chain object that manages the prompt, LLM, and session history.
        """

        chat_chain = (
            self.chat_prompt
            | self.llm
            | StrOutputParser()
        )
        chat_chain_with_message_history = RunnableWithMessageHistory(
            chat_chain,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        return chat_chain_with_message_history

    def get_response(self, user_input: str) -> str:
        """
        Generates a response based on the user's input and the current chat history.

        Args:
            user_input (str): The input message from the user.

        Returns:
            str: The generated response from the chat model.
        """
        if not self.chat_chain_with_message_history:
            self.chat_chain_with_message_history = self.create_chat_chain()

        response = self.chat_chain_with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": self.session_id}}
        )
        return response

    def clear_session_history(self) -> None:
        """
        Clears the chat message history for the current session.
        """
        if self.session_id in self.store:
            self.store[self.session_id].clear()
