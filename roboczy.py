from src.chat_manager import ChatManager
from src.vectorstore_manager import ChromaDatabaseManager
from src.information_extractor_manager import InformationExtractor
from src.retriever_manager import RAGManager
from src.prompt_manager import PromptManager
import chainlit as cl

extractor = InformationExtractor()
prompt_manager = PromptManager()


response = """ Here are two recommended plants that meet your requirements for vines, climbers, evergreen, wall climbers, and semi-shaded conditions:

Plant Name: Akebia quinata (Variety: Alba)

Description: A vigorous climber with semi-evergreen leaves composed of 5 leathery, light green leaflets. It produces fragrant white male and female flowers that bloom in April and May. This plant is known for its interesting fruit, resembling sausages, with edible flesh inside. It wraps around supports and is suitable for planting near fences, gazebos, pergolas, or gates.

Image Path: 124503300cf2670f2477c0251160c613db6f36b3.jpg

Plant Name: Akebia quinata (Variety: Silver Bells)

Description: This vigorous climber features semi-evergreen leaves made up of 5 leathery leaflets. It has fragrant male flowers that are white with pink-purple stamens and female flowers that are pink-purple with a dark chocolate pistil. The plant produces unique edible fruit and is suitable for planting on fences and various supports, thriving in semi-shaded locations.

Image Path: f1a3d937c2385a2ef39a
"""


dict_of_plants = extractor.extract_json(prompt_manager=prompt_manager, prompt_file_name="extract_dict", history=response)
print(dict_of_plants)
print(type(dict_of_plants))
