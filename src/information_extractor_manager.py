from typing import List, Union, Dict
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from src.prompt_manager import PromptManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformationExtractor:
    """
    Extracts information from responses using a OpenAI language model and output parsers.

    Attributes:
        llm (ChatOpenAI): Language model used for generating responses.
        prompt_manager (PromptManager): Instance of PromptManager used to manage prompt templates.
    """

    def __init__(self, prompt_manager: PromptManager, openai_llm_model_name: str = "gpt-4o-mini") -> None:
        """
        Initializes the InformationExtractor with the specified OpenAI model and a prompt manager.

        Args:
            openai_llm_model_name (str): The name of the OpenAI model to use. Defaults to "gpt-4o-mini".
            prompt_manager (PromptManager): Instance of PromptManager used to manage prompt templates.
        """
        self.llm = ChatOpenAI(model_name=openai_llm_model_name, temperature=0, max_tokens=256)
        self.prompt_manager = prompt_manager

    def _create_extract_chain(
            self,
            prompt: Union[PromptTemplate, ChatPromptTemplate],
            parser: Union[StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser]) \
            -> Runnable:
        """
        Creates a processing chain for extracting information based on the provided output parser.

        Args:
            prompt (Union [PromptTemplate, ChatPromptTemplate]): The prompt to be used for extraction.
            parser (Union[StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser]): The output parser to
            process the model's response, allowing for different types of output (string, list, or JSON).

        Returns:
            Runnable: A chain object that processes the prompt with the language model and the specified output parser.
        """
        extract_chain = (
            prompt
            | self.llm
            | parser
        )
        return extract_chain

    def extract_information_from_response(self, prompt_file_name: str, history: str) \
            -> str:
        """
        Extracts information from a response based on a prompt file and history, returning a cleaned string.

        Args:
            prompt_file_name (str): The name of the prompt file to load.
            history (str): The history of interactions to be passed to the prompt.

        Returns:
            str: The extracted information as a cleaned string.
        """
        prompt = self.prompt_manager.create_prompt(file_name=prompt_file_name)
        string_chain = self._create_extract_chain(prompt, parser=StrOutputParser())
        extracted_information = string_chain.invoke(history)
        clean_extracted_information = extracted_information.replace('\n', '').replace('"', '').strip()
        logger.info(f"Extracted {prompt_file_name}: {clean_extracted_information}")
        return clean_extracted_information

    def extract_plant_groups_from_response(self, prompt_file_name: str, history: str) \
            -> List[str]:
        """
    Extracts a list of plant groups from a response based on a prompt file and interaction history.
    This method invokes the extraction chain to retrieve plant groups. If no valid groups are found,
    it will attempt the extraction again. If the second attempt also yields no valid groups,
    it returns the complete list of available plant groups.

    Args:
        prompt_file_name (str): The name of the prompt file to load.
        history (str): The history of interactions to be passed to the prompt.

    Returns:
        List[str]: A list of extracted plant groups or the full list of available plant groups
                    if no valid groups are extracted.
        """
        available_plant_groups = ["perennial", "deciduous", "coniferous", "vines", "ericaceae", "fruity"]
        prompt = self.prompt_manager.create_prompt(file_name=prompt_file_name)
        plant_groups = self._create_extract_chain(prompt, parser=CommaSeparatedListOutputParser()).invoke(history)
        if not any(element in available_plant_groups for element in plant_groups):
            plant_groups = self._create_extract_chain(prompt, parser=CommaSeparatedListOutputParser()).invoke(history)
            if not any(element in available_plant_groups for element in plant_groups):
                return available_plant_groups
        logger.info(f"Extracted plant groups: {plant_groups}")
        return plant_groups

    def extract_plant_names_and_image_paths(self, prompt_file_name: str, history: str) \
            -> Dict[str, str]:
        """
        Extracts a dictionary of plant names and corresponding image paths from a response based on a prompt file and
        history.

        Args:
            prompt_file_name (str): The name of the prompt file to load.
            history (str): The history of interactions to be passed to the prompt.

        Returns:
            Dict[str, str]: A dictionary where keys are plant names and values are image paths.
        """
        prompt = self.prompt_manager.create_prompt(file_name=prompt_file_name)
        json_chain = self._create_extract_chain(prompt, parser=JsonOutputParser())
        dict_of_plants = json_chain.invoke(history)
        logger.info(f"Extracted plants from response: {dict_of_plants}")
        return dict_of_plants
