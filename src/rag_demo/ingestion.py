"""
ingestion.py
────────────
Fase de indexación del pipeline RAG.

Responsabilidades:
  1. Cargar documentos desde disco (PDF y TXT)
  2. Dividirlos en chunks con overlap
  3. Generar embeddings con nomic-embed-text via Ollama
  4. Guardar vectores + texto + metadata en ChromaDB

Este módulo se ejecuta UNA VEZ (o cuando añades documentos nuevos).
No interviene en las consultas del usuario.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_demo.config import settings

# ─── Logger ────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# __name__ es el nombre del módulo actual: "rag_demo.ingestion"
# Usar loggers en lugar de print() es una buena práctica:
#   - Puedes controlar el nivel (DEBUG, INFO, WARNING, ERROR)
#   - Puedes redirigir a archivos
#   - Puedes desactivarlos en producción sin tocar el código

# ─── Configuración del logging ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    # Ejemplo de output:
    # 2024-01-15 10:23:45 | INFO | rag_demo.ingestion | Cargando 3 documentos...
)


def load_documents(documents_path: Path) -> List[Document]:
    """
    Carga todos los PDFs y TXTs de una carpeta.

    Args:
        documents_path: ruta a la carpeta con documentos

    Returns:
        Lista de objetos Document de LangChain.
        Cada Document tiene:
          - .page_content: el texto extraído
          - .metadata: dict con fuente, página, etc.

    LangChain usa el tipo Document como unidad estándar
    para transportar texto + metadata a través del pipeline.
    """
    documents = []

    if not documents_path.exists():
        logger.warning(f"La carpeta {documents_path} no existe.")
        return documents

    # Buscar todos los archivos soportados
    # .glob() es un método de Path que busca archivos
    # por patrón. "**/*.pdf" significa:
    #   **   → en esta carpeta y todas las subcarpetas
    #   *    → cualquier nombre de archivo
    #   .pdf → con esta extensión
    pdf_files = list(documents_path.glob("**/*.pdf"))
    txt_files = list(documents_path.glob("**/*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        logger.warning(f"No se encontraron documentos en {documents_path}")
        return documents

    logger.info(f"Encontrados {len(all_files)} documentos: "
                f"{len(pdf_files)} PDFs, {len(txt_files)} TXTs")

    # Cargar cada archivo con el loader apropiado
    for file_path in all_files:
        try:
            if file_path.suffix.lower() == ".pdf":
                # PyPDFLoader: extrae texto página por página
                # Cada página se convierte en un Document separado
                # con metadata: {"source": "ruta.pdf", "page": 0}
                loader = PyPDFLoader(str(file_path))

            elif file_path.suffix.lower() == ".txt":
                # TextLoader: carga el archivo de texto completo
                # como un único Document
                # encoding="utf-8" para soportar caracteres especiales
                loader = TextLoader(str(file_path), encoding="utf-8")

            # .load() ejecuta la carga y devuelve List[Document]
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"  ✓ {file_path.name} → {len(docs)} páginas/secciones")

        except Exception as e:
            # Si un archivo falla, lo registramos pero continuamos
            # con los demás. Nunca dejamos que un archivo malo
            # rompa todo el proceso de indexación.
            logger.error(f"  ✗ Error cargando {file_path.name}: {e}")
            continue

    logger.info(f"Total: {len(documents)} secciones cargadas")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Divide los documentos en chunks más pequeños.

    Args:
        documents: lista de Documents cargados

    Returns:
        Lista de Documents divididos (chunks)

    Usamos RecursiveCharacterTextSplitter porque:
      1. Intenta dividir por párrafos primero (\n\n)
      2. Si el chunk sigue siendo grande, divide por líneas (\n)
      3. Luego por frases (". ")
      4. Como último recurso, por caracteres
      → Respeta la estructura natural del texto
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        # Tamaño máximo de cada chunk.
        # Se mide en caracteres, no tokens exactos.
        # Es una aproximación: 512 chars ≈ 128-170 tokens
        # dependiendo del texto.

        chunk_overlap=settings.chunk_overlap,
        # Caracteres compartidos entre chunks consecutivos.
        # Garantiza que la información en los bordes
        # no se pierda al cortar.

        separators=["\n\n", "\n", ". ", " ", ""],
        # Lista de separadores en orden de preferencia.
        # El splitter prueba cada uno hasta encontrar
        # uno que produzca chunks del tamaño correcto.

        length_function=len,
        # Función para medir el tamaño.
        # len() cuenta caracteres.
        # Alternativa: usar un tokenizer real,
        # pero para este proyecto len() es suficiente.

        add_start_index=True,
        # Añade a la metadata de cada chunk su posición
        # de inicio en el documento original.
        # Útil para debugging y para mostrar contexto.
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"Chunking: {len(documents)} secciones → {len(chunks)} chunks")
    logger.info(f"Configuración: chunk_size={settings.chunk_size}, "
                f"overlap={settings.chunk_overlap}")

    return chunks


def get_embedding_model() -> OllamaEmbeddings:
    """
    Crea y devuelve el modelo de embeddings.

    OllamaEmbeddings conecta con el servidor Ollama
    y usa nomic-embed-text para convertir texto en vectores.

    Returns:
        Instancia de OllamaEmbeddings lista para usar.
    """
    return OllamaEmbeddings(
        model=settings.embedding_model,
        # "nomic-embed-text" por defecto desde config.py

        base_url=settings.ollama_base_url,
        # "http://localhost:11434"
        # Ollama expone una API REST en este puerto.
        # OllamaEmbeddings hace llamadas HTTP a esta URL.
    )


def ingest_documents() -> int:
    """
    Función principal de indexación.

    Orquesta todo el pipeline:
      1. Carga documentos
      2. Divide en chunks
      3. Crea embeddings
      4. Guarda en ChromaDB

    Returns:
        Número de chunks indexados.

    Esta es la función que llama ingest.py (el script raíz).
    """
    logger.info("=" * 50)
    logger.info("Iniciando proceso de indexación")
    logger.info("=" * 50)

    # ── 1. Cargar documentos ───────────────────────────────────
    documents = load_documents(settings.documents_path)

    if not documents:
        logger.error("No hay documentos para indexar. "
                     f"Añade archivos en {settings.documents_path}")
        return 0

    # ── 2. Dividir en chunks ───────────────────────────────────
    chunks = split_documents(documents)

    if not chunks:
        logger.error("No se generaron chunks. Revisa los documentos.")
        return 0

    # ── 3. Crear modelo de embeddings ─────────────────────────
    logger.info(f"Usando embedding model: {settings.embedding_model}")
    embedding_model = get_embedding_model()

    # ── 4. Guardar en ChromaDB ─────────────────────────────────
    logger.info(f"Guardando en ChromaDB: {settings.chroma_path}")
    logger.info("Esto puede tardar varios minutos la primera vez...")
    logger.info("(Ollama está generando embeddings para cada chunk)")

    # Chroma.from_documents hace varias cosas a la vez:
    #   a) Para cada chunk, llama a embedding_model.embed_documents()
    #      → Ollama convierte el texto en un vector numérico
    #   b) Guarda el vector + el texto + la metadata en ChromaDB
    #   c) Persiste todo en disco (en chroma_persist_dir)
    #
    # Si ya existe la colección, la SOBREESCRIBE.
    # En un sistema productivo querrías lógica de upsert
    # (actualizar solo los documentos nuevos), pero para
    # aprender esto es suficiente y más claro.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
    )

    total = vector_store._collection.count()
    logger.info("=" * 50)
    logger.info(f"✅ Indexación completada: {total} chunks en ChromaDB")
    logger.info("=" * 50)

    return total