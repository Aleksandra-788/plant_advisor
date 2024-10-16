import pytest
from unittest.mock import MagicMock, patch
from src.information_extractor_manager import InformationExtractor
from src.prompt_manager import PromptManager
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser, JsonOutputParser
from langchain_core.runnables import Runnable
import sys
sys.path.append(".")


@pytest.fixture
@patch("src.information_extractor_manager.ChatOpenAI")
def information_extractor(mock_chat_openai):
    mock_chat_openai.return_value = MagicMock()
    return InformationExtractor(openai_llm_model_name="gpt-4o-mini")


class TestInformationExtractorManager:

    @pytest.fixture
    def mock_prompt_manager(self):
        return MagicMock(PromptManager)

    def test_create_extract_chain(self, information_extractor):
        prompt = ChatPromptTemplate.from_template("Test template")
        parser = StrOutputParser()
        string_chain = information_extractor._create_extract_chain(prompt, parser)
        assert isinstance(string_chain, Runnable)

    @patch("src.information_extractor_manager.PromptManager")
    def test_extract_informations_from_response(self, mock_prompt_manager, information_extractor):
        mock_string_chain = MagicMock()
        mock_string_chain.invoke.return_value = "Extracted info"
        information_extractor._create_extract_chain = MagicMock(return_value=mock_string_chain)
        result = information_extractor.extract_informations_from_response(mock_prompt_manager, "file_name", "history")
        assert result == "Extracted info"
        mock_string_chain.invoke.assert_called_once_with("history")

    @patch("src.information_extractor_manager.PromptManager")
    def test_extract_plant_groups_from_response_first_call(self, mock_prompt_manager, information_extractor):
        mock_list_chain = MagicMock()
        mock_list_chain.invoke.return_value = ["perennial", "deciduous"]
        information_extractor._create_extract_chain = MagicMock(return_value=mock_list_chain)
        result = information_extractor.extract_plant_groups_from_response(mock_prompt_manager, "file_name", "history")
        assert result == ["perennial", "deciduous"]
        mock_list_chain.invoke.assert_called_once_with("history")

    @patch("src.information_extractor_manager.PromptManager")
    def test_extract_plant_groups_from_response_multiple_calls(self, mock_prompt_manager, information_extractor):
        mock_list_chain = MagicMock()
        mock_list_chain.invoke.side_effect = [["unknown"], ["coniferous", "vines"]]
        information_extractor._create_extract_chain = MagicMock(return_value=mock_list_chain)
        result = information_extractor.extract_plant_groups_from_response(mock_prompt_manager, "file_name", "history")
        assert result == ["coniferous", "vines"]
        assert mock_list_chain.invoke.call_count == 2

    @patch("src.information_extractor_manager.PromptManager")
    def test_extract_plant_names_and_image_paths(self, mock_prompt_manager, information_extractor):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"tree": "12345.jpg", "flower": "234.jpg"}
        information_extractor._create_extract_chain = MagicMock(return_value=mock_chain)
        result = information_extractor.extract_plant_names_and_image_paths(mock_prompt_manager, "file_name", "history")
        assert result == {"tree": "12345.jpg", "flower": "234.jpg"}
        mock_chain.invoke.assert_called_once_with("history")
