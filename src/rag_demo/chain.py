"""
chain.py
────────
Construcción del RAG chain completo.

Responsabilidades:
  1. Definir el prompt template (system prompt + contexto + pregunta)
  2. Conectar retriever → prompt → LLM
  3. Exponer una función simple para invocar el chain

El "chain" es la abstracción de LangChain para una
secuencia de operaciones conectadas:

  pregunta → [retriever] → chunks
                        → [prompt template] → prompt completo
                                           → [LLM] → respuesta
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List

from rag_demo.config import settings
from rag_demo.retriever import get_retriever

logger = logging.getLogger(__name__)


def format_docs(docs: List[Document]) -> str:
    """
    Formatea los chunks recuperados en un string de contexto.

    Args:
        docs: lista de Documents devueltos por el retriever

    Returns:
        String con todos los chunks formateados,
        separados por líneas dobles, con su fuente.

    Este string es lo que va dentro del {context}
    del prompt template.

    Ejemplo de output:
      [Fuente: convenio.pdf | Página: 12]
      Los empleados tienen 25 días de vacaciones...

      [Fuente: convenio.pdf | Página: 13]
      Las vacaciones se solicitan con 15 días...
    """
    formatted = []
    for doc in docs:
        # Extraer metadata de fuente
        source = doc.metadata.get("source", "Desconocido")
        page = doc.metadata.get("page", "")

        # Construir header de fuente
        if page != "":
            header = f"[Fuente: {source} | Página: {int(page) + 1}]"
            # +1 porque PyPDFLoader empieza a contar páginas desde 0
        else:
            header = f"[Fuente: {source}]"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(formatted)
    # "\n\n" entre chunks para que el LLM los distinga claramente


# ─── Prompt Template ───────────────────────────────────────────
# Este es el prompt que recibirá el LLM en cada consulta.
# {context} y {question} son placeholders que LangChain
# rellena automáticamente antes de enviar al LLM.

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Eres un asistente farmacéutico experto. Responde preguntas \
usando la información del contexto proporcionado.

Instrucciones:
1. Usa PREFERENTEMENTE la información del contexto.
2. Si el contexto contiene información relevante aunque sea parcial, úsala.
3. Cita la fuente al final de tu respuesta: [Fuente: nombre_archivo]
4. Solo di "No encuentro información" si el contexto está completamente vacío
   o es totalmente irrelevante a la pregunta.
5. Sé claro y directo.

Contexto:
{context}"""
    ),
    (
        "human",
        "{question}"
    ),
])
# ChatPromptTemplate.from_messages acepta una lista de tuplas:
#   ("system", "...") → mensaje de sistema (instrucciones al LLM)
#   ("human", "...")  → mensaje del usuario


def get_llm() -> ChatOllama:
    """
    Crea la instancia del LLM.

    ChatOllama conecta con el servidor Ollama
    y expone una interfaz de chat estándar.

    Returns:
        Instancia de ChatOllama configurada.
    """
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,

        temperature=0.1,
        # Controla la "creatividad" del LLM.
        # 0.0 → completamente determinista (siempre la misma respuesta)
        # 1.0 → muy creativo/aleatorio
        # 0.1 → casi determinista, con poca variación
        # Para RAG queremos baja temperatura: el LLM debe
        # ceñirse al contexto, no inventar. 0.1 es ideal.
    )


def build_rag_chain():
    """
    Construye el RAG chain completo.

    El chain usa el operador | (pipe) de LangChain,
    que encadena operaciones de izquierda a derecha.

    Flujo:
      {"context": retriever, "question": passthrough}
           ↓
      RAG_PROMPT  (rellena {context} y {question})
           ↓
      LLM  (genera la respuesta)
           ↓
      StrOutputParser  (extrae el texto de la respuesta)

    Returns:
        Chain invocable. Uso:
          chain = build_rag_chain()
          respuesta = chain.invoke("¿Cuántos días de vacaciones?")
    """
    retriever = get_retriever()
    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs,
            # retriever recibe la pregunta → devuelve List[Document]
            # format_docs convierte esos docs en un string de contexto
            # ese string va al {context} del prompt

            "question": RunnablePassthrough(),
            # RunnablePassthrough pasa la pregunta tal cual
            # al {question} del prompt
            # Es como decir: "este valor no se transforma"
        }
        | RAG_PROMPT
        # El dict con context y question rellena el template
        # produciendo el prompt final (system + human)

        | llm
        # El prompt completo va al LLM
        # que genera un AIMessage con la respuesta

        | StrOutputParser()
        # Extrae el texto del AIMessage
        # Convierte AIMessage(content="...") → "..."
    )

    logger.info("RAG chain construido correctamente")
    return chain