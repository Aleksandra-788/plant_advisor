from typing import List, Union, Dict
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser, JsonOutputParser
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

    def _create_extract_chain(
            self,
            prompt: PromptTemplate,
            parser: Union[StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser]) \
            -> Runnable:
        """
        Creates a processing chain for extracting information based on the provided output parser.

        Args:
            prompt (PromptTemplate): The prompt to be used for extraction.
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
        string_chain = self._create_extract_chain(prompt, parser=StrOutputParser())
        extracted_informations = string_chain.invoke(history)
        clean_extracted_informations = extracted_informations.replace('\n', '').replace('"', '').strip()
        return clean_extracted_informations

    def extract_plant_groups_from_response(self, prompt_manager: PromptManager, prompt_file_name: str, history: str) \
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
        plant_groups = self._create_extract_chain(prompt, parser=CommaSeparatedListOutputParser()).invoke(history)
        if any(element in available_plant_groups for element in plant_groups):
            return plant_groups
        else:
            plant_groups = self._create_extract_chain(prompt, parser=CommaSeparatedListOutputParser()).invoke(history)
            return plant_groups

    def extract_plant_names_and_image_paths(self, prompt_manager: PromptManager, prompt_file_name: str, history: str) \
            -> Dict[str, str]:
        """
        Extracts a dictionary of plant names and corresponding image paths from a response based on a prompt file and
        history.

        Args:
            prompt_manager (PromptManager): An instance of PromptManager used to create the prompt.
            prompt_file_name (str): The name of the prompt file to load.
            history (str): The history of interactions to be passed to the prompt.

        Returns:
            Dict[str, str]: A dictionary where keys are plant names and values are image paths.
        """
        prompt = prompt_manager.create_prompt(file_name=prompt_file_name)
        json_chain = self._create_extract_chain(prompt, parser=JsonOutputParser())
        dict_of_plants = json_chain.invoke(history)
        return dict_of_plants
