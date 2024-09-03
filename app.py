import logging
from src.chat_manager import ChatManager
from src.vectorstore_manager import ChromaDatabaseManager
from src.information_extractor_manager import InformationExtractor
from src.retriever_manager import RAGManager
from src.prompt_manager import PromptManager
import chainlit as cl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

extractor = InformationExtractor()
prompt_manager = PromptManager()
chat_manager = ChatManager(prompt_manager=prompt_manager)
database_manager = ChromaDatabaseManager()

logger.info("Checking if database collection exists and is filled...")
if not database_manager.collection_exists_and_filled():
    logger.info("Creating embedding vectorstore...")
    database_manager.create_embedding_vectorstore()
else:
    logger.info("Collection already exists and is filled with data.")


@cl.on_chat_start
async def on_chat_start():
    logger.info("Chat started. Initializing session variables.")
    database = database_manager.create_chroma_database()
    history = chat_manager.get_session_history()

    cl.user_session.set("database", database)
    cl.user_session.set("history", history)

    logger.info("Sending welcome message to user.")
    await cl.Message(content="I am a virtual gardening assistant. "
                             "I will help you choose the perfect plant for your garden. "
                             "Just help me determine what you’re looking for. "
                             "Let’s get started!").send()


@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received message from user: {message.content}")

    database = cl.user_session.get("database")
    history = cl.user_session.get("history")

    response = chat_manager.get_response(user_input=message.content)
    logger.info(f"Generated response: {response}")

    if conversation_ended(response):
        logger.info("Conversation ended detected. Handling conversation end.")
        await handle_conversation_end(history, database)
        await end()

    else:
        logger.info("Sending response to user.")
        await cl.Message(content=response).send()


@cl.on_chat_end
async def end():
    logger.info("Chat ended. Sending thank you message.")
    await cl.Message(content="Thank you for conversation!").send()


def conversation_ended(response):
    ended = "END OF CONVERSATION" in response
    logger.info(f"Checking if conversation ended: {ended}")
    return ended


async def handle_conversation_end(history, database):
    logger.info("Extracting plant description from conversation history.")
    plant_description = extractor.extract_informations_from_response(prompt_manager=prompt_manager,
                                                                     prompt_file_name="extract_informations_template",
                                                                   history=history)
    logger.info(f"Extracted plant description: {plant_description}")
    logger.info("Extracting plant groups from conversation history.")
    plant_groups = extractor.extract_plant_groups_from_response(prompt_manager=prompt_manager,
                                                        prompt_file_name="extract_plant_group_template",
                                                        history=history)
    logger.info(f"Extracted plant groups: {plant_groups}")
    rag_manager = RAGManager(prompt_manager=prompt_manager, vectorstore=database)

    # debug
    logger.info("Creating retriever and invoking document retrieval.")
    retriever = rag_manager.create_retriever(plant_groups=plant_groups)
    retriever_docs = retriever.invoke(plant_description)
    logger.info(f"Retrieved documents: {[doc.page_content for doc in retriever_docs]}")

    logger.info("Generating response based on retrieved documents.")
    response = rag_manager.get_response(plant_groups=plant_groups, plant_description=plant_description)
    logger.info(f"Response: {response}")

    dict_of_plants = extractor.extract_json(prompt_manager=prompt_manager, prompt_file_name="extract_dict",
                                            history=response)
    logger.info(f"Extracted plants from response: {dict_of_plants}")

    final_response = extractor.extract_informations_from_response(prompt_manager=prompt_manager,
                                                                  prompt_file_name="extract_final_response",
                                                                  history=response)
    logger.info(f"Final response generated: {final_response}")
    chat_manager.clear_session_history()
    logger.info("Cleared session history.")
    await cl.Message(content=final_response).send()
    for plant_name, image_path in dict_of_plants.items():
        image = cl.Image(path=f"data/images/{image_path}", display='inline', size='large', name='plant')
        await cl.Message(
            content=plant_name,
            elements=[image],
        ).send()



