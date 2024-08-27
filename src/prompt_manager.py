from typing import Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
import os


class PromptManager:
    def __init__(self, template_directory: str = "prompt_templates/"):
        """
        Initializes the PromptManager with the directory of prompt templates and a template file name.

        Args:
            template_directory (str): The directory where the prompt templates are stored.
            Defaults to "prompt_templates/".
        """
        self.template_directory = template_directory

    def _load_template(self, file_name) -> str:
        """
        Loads the prompt template from a file.

        Returns:
            str: The content of the template file as a string.

        """
        full_path = os.path.join(self.template_directory, file_name)
        with open(full_path, 'r') as file:
            return file.read()

    def create_prompt(self, file_name) -> Union[ChatPromptTemplate, PromptTemplate]:
        """
        Creates a prompt object based on the specified template file.

        If the file name is "conversational_template", it creates a ChatPromptTemplate with a system message,
        a placeholder for message history, and a human input message. For other file names, it creates a standard
        PromptTemplate.

        Returns:
            Union[ChatPromptTemplate, PromptTemplate]: The prompt object created from the template file.

        """

        if file_name == "conversational_template":
            conversational_template = self._load_template(file_name)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", conversational_template),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
            )
        else:
            template = self._load_template(file_name)
            prompt = PromptTemplate.from_template(template)
        return prompt
