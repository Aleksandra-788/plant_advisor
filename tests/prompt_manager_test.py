import pytest
from unittest.mock import patch, mock_open
from src.prompt_manager import PromptManager
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
import sys
sys.path.append(".")


class TestPromptManager:

    @pytest.fixture
    def prompt_manager(self):
        return PromptManager(template_directory="test_templates/")

    @patch("builtins.open", new_callable=mock_open, read_data="Test template content")
    def test_load_template(self, mock_file, prompt_manager):
        template_content = prompt_manager._load_template("test_template.txt")
        assert template_content == "Test template content"
        mock_file.assert_called_once_with("test_templates/test_template.txt", 'r')

    @patch("builtins.open", new_callable=mock_open, read_data="Conversational template content")
    def test_create_prompt_conversational(self, mock_file, prompt_manager):
        prompt = prompt_manager.create_prompt("conversational_template")
        assert isinstance(prompt, ChatPromptTemplate)
        system_message_templates = [msg for msg in prompt.messages if isinstance(msg, SystemMessagePromptTemplate)]
        assert len(system_message_templates) == 1
        assert "Conversational template content" in system_message_templates[0].prompt.template

        mock_file.assert_called_once_with("test_templates/conversational_template", 'r')

    @patch("builtins.open", new_callable=mock_open, read_data="Standard template content")
    def test_create_prompt_standard(self, mock_file, prompt_manager):
        prompt = prompt_manager.create_prompt("standard_template.txt")
        assert isinstance(prompt, PromptTemplate)
        assert "Standard template content" in prompt.template
        mock_file.assert_called_once_with("test_templates/standard_template.txt", 'r')

    @patch("builtins.open", new_callable=mock_open)
    def test_load_template_file_not_found(self, mock_file, prompt_manager):
        mock_file.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            prompt_manager._load_template("non_existent_template.txt")
        mock_file.assert_called_once_with("test_templates/non_existent_template.txt", 'r')
