"""
ingest.py
─────────
Script de entrada para indexar documentos.

Uso:
    uv run ingest.py
"""

import logging
import sys

# Ahora importamos directamente desde rag_demo (sin src.)
# porque el paquete está instalado en el entorno virtual
from rag_demo.ingestion import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("🚀 Iniciando indexación de documentos...")

    try:
        total = ingest_documents()

        if total == 0:
            logger.error("❌ No se indexó ningún documento.")
            logger.error("   Asegúrate de tener archivos en data/documents/")
            sys.exit(1)

        logger.info(f"✅ Listo. {total} chunks indexados.")
        logger.info("   Ya puedes arrancar la app: chainlit run app.py")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Error durante la indexación: {e}")
        logger.exception(e)
        sys.exit(1)