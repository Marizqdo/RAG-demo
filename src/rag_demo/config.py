"""
config.py
─────────
Configuración centralizada del proyecto RAG.

Usa Pydantic Settings para:
  1. Leer variables desde el archivo .env
  2. Validar que los tipos son correctos
  3. Proporcionar valores por defecto
  4. Exponer un único objeto `settings` importable desde cualquier módulo
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Clase de configuración.

    Pydantic Settings lee automáticamente las variables
    del archivo .env y las mapea a estos atributos.

    El orden de prioridad es:
      1. Variables de entorno del sistema (export VAR=valor)
      2. Variables del archivo .env
      3. Valores por defecto definidos aquí
    """

    model_config = SettingsConfigDict(
        # Le decimos a Pydantic dónde está el archivo .env
        env_file=".env",
        # Si hay variables en .env que no están en esta clase,
        # las ignora en lugar de lanzar error
        extra="ignore",
    )

    # ─── Ollama ────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    # URL donde escucha el servidor Ollama.
    # localhost porque corre en Docker con puerto mapeado al host.

    # ─── Modelos ───────────────────────────────────────────────
    llm_model: str = "llama3.2"
    # El modelo LLM que usará Ollama para generar respuestas.
    # Intercambiable: podrías poner "mistral", "phi3", etc.

    embedding_model: str = "nomic-embed-text"
    # El modelo que convierte texto en vectores.
    # nomic-embed-text genera vectores de 768 dimensiones.

    # ─── ChromaDB ──────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    # Carpeta donde ChromaDB guarda los vectores en disco.
    # Si no existe, ChromaDB la crea automáticamente.

    chroma_collection_name: str = "rag_documents"
    # Nombre de la "colección" dentro de ChromaDB.
    # Una colección es como una tabla en SQL:
    # agrupa vectores relacionados.

    # ─── Chunking ──────────────────────────────────────────────
    chunk_size: int = 512
    # Tamaño máximo de cada chunk en tokens (aproximado).
    # 512 es un buen punto de partida para docs técnicos.
    # Más pequeño → más precisión, menos contexto por chunk.
    # Más grande  → más contexto, menos precisión de retrieval.

    chunk_overlap: int = 64
    # Tokens compartidos entre chunks consecutivos.
    # 64 sobre 512 = ~12.5% de overlap.
    # Evita perder información en los cortes.

    # ─── Retrieval ─────────────────────────────────────────────
    top_k_results: int = 4
    # Número de chunks a recuperar por consulta.
    # 4 es un buen balance: suficiente contexto,
    # sin saturar el prompt del LLM.

    # ─── Documentos ────────────────────────────────────────────
    documents_dir: str = "./data/documents"
    # Carpeta donde el usuario pone sus PDFs y TXTs.

    @property
    def documents_path(self) -> Path:
        """
        Devuelve la ruta de documentos como objeto Path.

        Path es mejor que str para rutas:
          - Path("./data") / "documents"  en lugar de
            "./data" + "/" + "documents"
          - Funciona igual en Linux, Mac y Windows
          - Tiene métodos útiles: .exists(), .glob(), .stem...
        """
        return Path(self.documents_dir)

    @property
    def chroma_path(self) -> Path:
        """Devuelve la ruta de ChromaDB como objeto Path."""
        return Path(self.chroma_persist_dir)


# ─── Instancia global ──────────────────────────────────────────
settings = Settings()
# Creamos UNA sola instancia de Settings.
# Todos los módulos importan este objeto:
#
#   from rag_demo.config import settings
#   print(settings.chunk_size)  # → 512
#
# Pydantic lee el .env en este momento (al importar).