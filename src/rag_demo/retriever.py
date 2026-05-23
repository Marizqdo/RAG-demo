"""
retriever.py
────────────
Recuperación de chunks relevantes desde ChromaDB.

Responsabilidades:
  1. Conectar con ChromaDB existente (ya indexado)
  2. Dado un texto de consulta, encontrar los chunks
     más semánticamente similares
  3. Devolver esos chunks para construir el prompt

Este módulo se usa en CADA consulta del usuario.
"""

import logging
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from rag_demo.config import settings

logger = logging.getLogger(__name__)


def get_vector_store() -> Chroma:
    """
    Conecta con el vector store ChromaDB existente.

    A diferencia de Chroma.from_documents() (que CREA),
    aquí usamos el constructor directo de Chroma (que CONECTA).

    No genera embeddings aquí: solo abre la conexión
    con la base de datos ya indexada en disco.

    Returns:
        Instancia de Chroma conectada a la BD existente.

    Raises:
        Exception si ChromaDB no existe o está vacía.
        (El usuario debe ejecutar ingest.py primero)
    """
    embedding_model = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        # Necesitamos el modelo de embeddings aquí también:
        # cuando llega una pregunta del usuario, hay que
        # convertirla a vector ANTES de buscar en ChromaDB.
        # El modelo debe ser EL MISMO que se usó en ingestion.py.
        # Si indexaste con nomic-embed-text y buscas con
        # otro modelo → los vectores son incompatibles → resultados basura.
    )

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_persist_dir,
    )

    # Verificar que hay datos indexados
    count = vector_store._collection.count()
    if count == 0:
        raise ValueError(
            "ChromaDB está vacía. "
            "Ejecuta primero: uv run ingest.py"
        )

    logger.info(f"ChromaDB conectada: {count} chunks disponibles")
    return vector_store


def get_retriever() -> VectorStoreRetriever:
    """
    Crea el retriever a partir del vector store.

    Un retriever es la abstracción de LangChain para
    "dado un texto, devuelve documentos relevantes".

    La diferencia entre VectorStore y Retriever:
      VectorStore → base de datos (almacena y busca vectores)
      Retriever   → interfaz de búsqueda (devuelve Documents)

    El retriever es lo que se conecta al chain en chain.py.

    Returns:
        VectorStoreRetriever configurado.
    """
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        # "similarity" → búsqueda por similitud coseno (ANN)
        # Alternativas:
        #   "mmr" → Maximum Marginal Relevance
        #           devuelve chunks relevantes Y diversos
        #           evita chunks muy parecidos entre sí
        #   "similarity_score_threshold"
        #           solo devuelve chunks con score > umbral

        search_kwargs={
            "k": settings.top_k_results,
            # Número de chunks a devolver (top-4 por defecto)
        },
    )

    logger.info(f"Retriever creado: top_k={settings.top_k_results}, "
                f"tipo=similarity")
    return retriever