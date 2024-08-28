from typing import List, Union
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from src.prompt_manager import PromptManager


class InformationExtractor:
    """
    Extracts information from responses using a language model and output parsers.

    Attributes:
        llm (ChatOpenAI): Language model used for generating responses.
    """

    def __init__(self, openai_llm_model_name: str = "gpt-4o-mini") -> None:
        """
        Initializes the InformationExtractor with the specified OpenAI model.

        Args:
            openai_llm_model_name (str): The name of the OpenAI model to use. Defaults to "gpt-4o-mini".
        """
        self.llm = ChatOpenAI(model_name=openai_llm_model_name, temperature=0, max_tokens=256)

    def _create_extract_string_chain(self, prompt: Union[ChatPromptTemplate, PromptTemplate]) -> Runnable:
        """
        Creates a processing chain for extracting information as a single string.

        Args:
            prompt (Union[ChatPromptTemplate, PromptTemplate]): The prompt to be used for extraction.

        Returns:
            Runnable: A chain object that processes the prompt with the language model and string output parser.
        """
        extract_string_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        return extract_string_chain

    def _create_extract_list_chain(self, prompt: PromptTemplate) -> Runnable:
        """
        Creates a processing chain for extracting information as a comma-separated list.

        Args:
            prompt (PromptTemplate): The prompt to be used for extraction.

        Returns:
            Runnable: A chain object that processes the prompt with the language model and comma-separated list output
            parser.
        """
        extract_list_chain = (
            prompt
            | self.llm
            | CommaSeparatedListOutputParser()
        )
        return extract_list_chain

    def extract_informations_from_response(self, prompt_manager: PromptManager, prompt_file_name: str, history: str) \
            -> str:
        """
        Extracts information from a response based on a prompt file and history, returning a cleaned string.

        Args:
            prompt_manager (PromptManager): An instance of PromptManager used to create the prompt.
            prompt_file_name (str): The name of the prompt file to load.
            history (str): The history of interactions to be passed to the prompt.

        Returns:
            str: The extracted information as a cleaned string.
        """
        prompt = prompt_manager.create_prompt(file_name=prompt_file_name)
        string_chain = self._create_extract_string_chain(prompt)
        extracted_informations = string_chain.invoke(history)
        clean_extracted_informations = extracted_informations.replace('\n', '').replace('"', '').strip()
        return clean_extracted_informations

    def extract_list_from_response(self, prompt_manager: PromptManager, prompt_file_name: str, history: str) \
            -> List[str]:
        """
        Extracts a list of elements from a response based on a prompt file and history.

        Args:
            prompt_manager (PromptManager): An instance of PromptManager used to create the prompt.
            prompt_file_name (str): The name of the prompt file to load.
            history (str): The history of interactions to be passed to the prompt.

        Returns:
            List[str]: The extracted list of elements.
        """
        available_plant_groups = ["perennial", "deciduous", "coniferous", "vines", "ericaceae", "fruity"]
        prompt = prompt_manager.create_prompt(file_name=prompt_file_name)
        list_of_elements = self._create_extract_list_chain(prompt).invoke(history)
        if any(element in available_plant_groups for element in list_of_elements):
            return list_of_elements
        else:
            list_of_elements = self._create_extract_list_chain(prompt).invoke(history)
            return list_of_elements
