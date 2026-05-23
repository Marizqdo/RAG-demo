import chainlit as cl
from langchain_core.documents import Document
from typing import List

# Sin prefijo src.
from rag_demo.chain import build_rag_chain
from rag_demo.retriever import get_vector_store

@cl.on_chat_start
async def on_chat_start():
    try:
        chain = build_rag_chain()
        cl.user_session.set("chain", chain)

        await cl.Message(
            content=(
                "👋 ¡Hola! Soy tu asistente RAG.\n\n"
                "Puedo responder preguntas sobre los documentos indexados. "
                "¿Qué quieres saber?"
            )
        ).send()

    except ValueError as e:
        await cl.Message(
            content=(
                f"⚠️ Error al iniciar: {e}\n\n"
                "Por favor ejecuta primero:\n"
                "```bash\nuv run ingest.py\n```"
            )
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")

    if not chain:
        await cl.Message(
            content="❌ El sistema no está inicializado. Recarga la página."
        ).send()
        return

    question = message.content

    try:
        vector_store = get_vector_store()
        relevant_docs: List[Document] = vector_store.similarity_search(
            question, k=4
        )
    except Exception:
        relevant_docs = []

    response_message = cl.Message(content="")
    await response_message.send()

    full_response = ""
    async for chunk in chain.astream(question):
        await response_message.stream_token(chunk)
        full_response += chunk

    if relevant_docs:
        source_elements = []

        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source", "Desconocido")
            page = doc.metadata.get("page", "")
            source_name = f"Fuente {i+1}"

            source_elements.append(
                cl.Text(
                    name=source_name,
                    content=(
                        f"📄 {source}"
                        + (f" (pág. {int(page)+1})" if page != "" else "")
                        + f"\n\n{doc.page_content}"
                    ),
                    display="inline",
                )
            )

        response_message.elements = source_elements
        await response_message.update()