from src.chat_manager import ChatManager
from src.vectorstore_manager import ChromaDatabaseManager
from src.information_extractor_manager import InformationExtractor
from src.retriever_manager import RAGManager
from src.prompt_manager import PromptManager
import chainlit as cl


extractor = InformationExtractor()
prompt_manager = PromptManager()
chat_manager = ChatManager(prompt_manager=prompt_manager)
database_manager = ChromaDatabaseManager()

if not database_manager.collection_exists_and_filled():
    database_manager.create_embedding_vectorstore()
else:
    print("Collection already exists and is filled with data.")


@cl.on_chat_start
async def on_chat_start():
    database = database_manager.create_chroma_database()
    history = chat_manager.get_session_history()

    cl.user_session.set("database", database)
    cl.user_session.set("history", history)

    await cl.Message(content="I am a virtual gardening advisor. How can I help you!").send()


@cl.on_message
async def main(message: cl.Message):

    database = cl.user_session.get("database")
    history = cl.user_session.get("history")

    response = chat_manager.get_response(user_input=message.content)

    if conversation_ended(response):
        await handle_conversation_end(history, database)
        await end()

    else:
        await cl.Message(content=response).send()


@cl.on_chat_end
async def end():
    await cl.Message(content="Thank you for conversation!").send()


def conversation_ended(response):
    return "END OF CONVERSATION" in response


async def handle_conversation_end(history, database):

    plant_description = extractor.extract_informations_from_response(prompt_manager=prompt_manager,
                                                                     prompt_file_name="extract_informations_template",
                                                                     history=history)
    plant_groups = extractor.extract_list_from_response(prompt_manager=prompt_manager,
                                                        prompt_file_name="extract_plant_group_template",
                                                        history=history)
    rag_manager = RAGManager(prompt_manager=prompt_manager, vectorstore=database)

    # debug
    retriever = rag_manager.create_retriever(plant_groups=plant_groups)
    # retriever_docs = retriever.get_relevant_documents(plant_description)
    retriever_docs = retriever.invoke(plant_description)
    response = rag_manager.get_response(plant_groups=plant_groups, plant_description=plant_description)
    final_response = extractor.extract_informations_from_response(prompt_manager=prompt_manager,
                                                                  prompt_file_name="extract_final_response",
                                                                  history=response)

    image_name = extractor.extract_informations_from_response(prompt_manager=prompt_manager,
                                                              prompt_file_name="extract_image_path", history=response)
    chat_manager.clear_session_history()

    await cl.Message(content=f"Plant Description: {plant_description}").send()
    await cl.Message(content=f"Plant Groups: {plant_groups}").send()
    for idx, doc in enumerate(retriever_docs):
        await cl.Message(content=f"Document {idx + 1}: {doc.page_content}").send()
    await cl.Message(content=f"Response: {response}").send()
    await cl.Message(content=f"Final Response: {final_response}").send()
    await cl.Message(content=f"Image Path: {image_name}").send()

    if image_name != "9a58e99369a6799c9fc054c69935d7a509541b9e.jpg":
        image = cl.Image(path=f"data/images/{image_name}", display='inline', size='large', name='plant')
        await cl.Message(
            content="Plant image",
            elements=[image],
        ).send()
