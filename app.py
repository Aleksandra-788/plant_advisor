import logging
from src.chat_manager import ChatManager
from src.chroma_database_manager import ChromaDatabaseManager
from src.information_extractor_manager import InformationExtractor
from src.RAG_manager import RAGManager
from src.prompt_manager import PromptManager
import chainlit as cl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt_manager = PromptManager()
extractor = InformationExtractor(prompt_manager=prompt_manager)
chat_manager = ChatManager(prompt_manager=prompt_manager)
database_manager = ChromaDatabaseManager()
database = database_manager.database
history = chat_manager.get_session_history()
rag_manager = RAGManager(prompt_manager=prompt_manager, vectorstore=database)


@cl.on_chat_start
async def on_chat_start():
    """
        Initialize the chat session, set session variables, and send a welcome message.
    """
    logger.info("Chat started. Initializing session variables.")
    await cl.Message(content="I am a virtual gardening assistant. "
                             "I will help you choose the perfect plant for your garden. "
                             "Just help me determine what you’re looking for. "
                             "Let’s get started!").send()


@cl.on_message
async def main(message: cl.Message):
    """
        Handle incoming messages from the user, generate a response, and check if the conversation has ended.
    """
    logger.info(f"Received message from user: {message.content}")
    response = chat_manager.get_response(user_input=message.content)
    if chat_manager.conversation_ended(response):
        await handle_conversation_end()
        await end()
    else:
        logger.info("Sending response to user.")
        await cl.Message(content=response).send()


@cl.on_chat_end
async def end():
    """
        Send a thank you message when the chat ends.
    """
    logger.info("Chat ended. Sending thank you message.")
    await cl.Message(content="Thank you for conversation!").send()


async def handle_conversation_end():
    """
        Handle the end of the conversation by extracting information, retrieving documents,
        generating a response, and sending plant details.
    """
    plant_description = extractor.extract_information_from_response(prompt_file_name="extract_plant_description",
                                                                    history=history)
    plant_groups = extractor.extract_plant_groups_from_response(prompt_file_name="extract_plant_group",
                                                                history=history)
    rag_manager.retrieve_documents(plant_groups=plant_groups, plant_description=plant_description)
    response = rag_manager.get_response(plant_groups=plant_groups, plant_description=plant_description)
    dict_of_plants = extractor.extract_plant_names_and_image_paths(prompt_file_name="extract_plant_name_image_path",
                                                                   history=response)
    final_response = extractor.extract_information_from_response(prompt_file_name="extract_final_response",
                                                                 history=response)
    chat_manager.clear_session_history()
    await cl.Message(content=final_response).send()
    for plant_name, image_path in dict_of_plants.items():
        if image_path != "9a58e99369a6799c9fc054c69935d7a509541b9e.jpg":
            image = cl.Image(path=f"data/images/{image_path}", display='inline', size='large', name='plant')
            await cl.Message(
                content=plant_name,
                elements=[image],
            ).send()
