# 🔍 RAG Demo — Retrieval-Augmented Generation con LangChain, Ollama y ChromaDB

> Sistema RAG completo, local y gratuito. Sin API keys. Sin coste por consulta. Sin datos que salen de tu máquina.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Ollama](https://img.shields.io/badge/Ollama-local-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-local-purple)
![Chainlit](https://img.shields.io/badge/Chainlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ¿Qué es este proyecto?

Este repositorio implementa un sistema **RAG (Retrieval-Augmented Generation)** de extremo a extremo, pensado como recurso educativo y punto de partida para proyectos reales.

RAG resuelve el problema más crítico de los LLMs en entornos empresariales:

```
❌ Sin RAG:  el LLM responde con su conocimiento general
             → alucina datos, ignora tu documentación interna

✅ Con RAG:  el LLM busca primero en tus documentos
             → responde con información real y verificable
             → cita las fuentes exactas de cada respuesta
```

---

## Arquitectura

```
                    ── FASE DE INDEXACIÓN (una vez) ──

  [PDFs / TXTs]
       │
       ▼
  [PyPDFLoader / TextLoader]     carga y extrae texto
       │
       ▼
  [RecursiveCharacterTextSplitter]  divide en chunks (512 tokens, overlap 64)
       │
       ▼
  [nomic-embed-text via Ollama]   texto → vector [768 dimensiones]
       │
       ▼
  [ChromaDB]                     almacena vector + texto + metadata en disco


                    ── FASE DE CONSULTA (en tiempo real) ──

  [Pregunta del usuario]
       │
       ▼
  [nomic-embed-text via Ollama]   pregunta → vector
       │
       ▼
  [ChromaDB — ANN search]        devuelve top-4 chunks más similares
       │
       ▼
  [Prompt augmentation]          system prompt + contexto + pregunta
       │
       ▼
  [Llama 3.2 via Ollama]         genera respuesta anclada al contexto
       │
       ▼
  [Chainlit UI]                  muestra respuesta + fuentes expandibles
```

---

## Stack tecnológico

| Componente | Tecnología | Por qué |
|---|---|---|
| Gestor de entorno | `uv` | Estándar moderno 2025/2026, rápido y reproducible |
| LLM | `Ollama + Llama 3.2` | Local, gratuito, sin API key |
| Embeddings | `nomic-embed-text` | Open source, local, calidad buena |
| Vector DB | `ChromaDB` | Sin servidor, persistente en disco, cero fricción |
| RAG Framework | `LangChain` | Abstracciones claras, bien documentado |
| UI de chat | `Chainlit` | Nativo para LLMs: streaming, fuentes, historial |
| Contenedor | `Docker + Compose` | Reproducible, aislado, buena práctica |

---

## Requisitos previos

- **Linux / macOS** (en Windows usar WSL2)
- **Python 3.12**
- **Docker** con Docker Compose plugin
- **uv** — gestor de paquetes Python moderno
- **RAM mínima**: 8 GB (para Llama 3.2 3B)
- **Espacio en disco**: ~4 GB (modelos de Ollama)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/rag-demo.git
cd rag-demo
```

### 2. Instalar uv (si no lo tienes)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3. Crear el entorno virtual e instalar dependencias

```bash
uv venv
uv sync
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# El archivo .env por defecto funciona sin cambios
# para una instalación local estándar
```

### 5. Levantar Ollama con Docker

```bash
docker compose up -d

# Verificar que está corriendo
curl http://localhost:11434
# Respuesta esperada: "Ollama is running"
```

### 6. Descargar los modelos

```bash
# Modelo de embeddings (~274 MB)
docker exec ollama ollama pull nomic-embed-text

# LLM — elige según tu RAM disponible:
docker exec ollama ollama pull llama3.2      # 3B — recomendado (8 GB RAM)
docker exec ollama ollama pull llama3.2:8b   # 8B — mejor calidad (16 GB RAM)
```

### 7. Añadir tus documentos

Copia tus archivos PDF o TXT en la carpeta `data/documents/`:

```bash
cp mis_documentos/*.pdf data/documents/
```

### 8. Indexar los documentos

```bash
uv run ingest.py
```

Verás el progreso en consola. Al terminar tendrás los vectores en `chroma_db/`.

### 9. Arrancar la aplicación

```bash
uv run chainlit run app.py --port 8000
```

Abre el navegador en **http://localhost:8000** y empieza a hacer preguntas.

---

## Uso

### Flujo básico

```
1. Añade documentos a data/documents/
2. Ejecuta: uv run ingest.py
3. Ejecuta: uv run chainlit run app.py --port 8000
4. Abre http://localhost:8000
5. Haz preguntas sobre tus documentos
```

### Actualizar documentos

Cada vez que añadas o cambies documentos, hay que reindexar:

```bash
# Limpiar índice anterior
rm -rf chroma_db/

# Reindexar
uv run ingest.py
```

### Arranque rápido (sesiones posteriores)

```bash
docker compose up -d
uv run chainlit run app.py --port 8000
```

---

## Estructura del proyecto

```
rag-demo/
│
├── data/
│   └── documents/          ← pon aquí tus PDFs y TXTs
│
├── src/
│   └── rag_demo/
│       ├── __init__.py
│       ├── config.py       ← configuración centralizada
│       ├── ingestion.py    ← carga, chunking y embeddings
│       ├── retriever.py    ← búsqueda semántica en ChromaDB
│       └── chain.py        ← prompt template y RAG chain
│
├── app.py                  ← interfaz Chainlit (punto de entrada)
├── ingest.py               ← script de indexación
├── docker-compose.yml      ← Ollama en Docker
├── pyproject.toml          ← dependencias (uv)
├── .env.example            ← plantilla de configuración
└── README.md
```

---

## Configuración

Todas las opciones están en `.env`. Los valores por defecto funcionan para una instalación local estándar:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Modelos
LLM_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=rag_documents

# Chunking — ajusta según tus documentos
CHUNK_SIZE=512        # tamaño de cada fragmento (caracteres)
CHUNK_OVERLAP=64      # solapamiento entre fragmentos

# Retrieval
TOP_K_RESULTS=4       # chunks a recuperar por consulta
```

### Guía de ajuste de parámetros

| Parámetro | Valor bajo | Valor alto | Recomendación |
|---|---|---|---|
| `CHUNK_SIZE` | más precisión, menos contexto | menos precisión, más contexto | 256-512 para FAQs; 512-1024 para docs técnicos |
| `CHUNK_OVERLAP` | riesgo de perder info en cortes | redundancia | 10-20% del chunk_size |
| `TOP_K_RESULTS` | respuestas más focalizadas | más contexto, más coste | 3-5 es el rango óptimo |

---

## Solución de problemas

**`Connection refused` al arrancar la app**
```bash
# Ollama no está corriendo
docker compose up -d
curl http://localhost:11434
```

**`ChromaDB está vacía`**
```bash
# No has ejecutado la indexación
uv run ingest.py
```

**`model not found`**
```bash
# El modelo no está descargado en Ollama
docker exec ollama ollama list
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull nomic-embed-text
```

**`ModuleNotFoundError`**
```bash
# Dependencias no instaladas
uv sync
```

**Respuestas de baja calidad o irrelevantes**
```bash
# Posibles causas:
# 1. Chunks demasiado grandes o pequeños → ajusta CHUNK_SIZE en .env
# 2. Poco overlap → aumenta CHUNK_OVERLAP
# 3. TOP_K muy bajo → sube a 5 o 6
# 4. PDFs con texto mal extraído (escaneados) → usa OCR primero
```

---

## Conceptos clave

Si quieres entender qué hace cada parte del sistema:

| Concepto | Descripción |
|---|---|
| **Embedding** | Representación numérica del significado de un texto. Textos similares tienen vectores cercanos. |
| **Vector DB** | Base de datos optimizada para buscar vectores por similitud semántica, no por coincidencia exacta. |
| **Chunking** | División de documentos en fragmentos manejables. El tamaño afecta directamente la calidad del retrieval. |
| **Retriever** | Componente que, dada una pregunta, busca los fragmentos más relevantes en la Vector DB. |
| **Prompt augmentation** | Técnica de añadir contexto recuperado al prompt antes de enviarlo al LLM. |
| **RAG** | Arquitectura que combina retrieval (búsqueda) con generation (LLM) para respuestas fundamentadas. |

---

## Limitaciones conocidas

- **PDFs escaneados**: el sistema extrae texto digital. Los PDFs que son imágenes escaneadas requieren OCR previo.
- **Razonamiento multi-hop**: preguntas que requieren combinar muchos fragmentos distantes pueden dar resultados subóptimos.
- **Idiomas**: nomic-embed-text funciona bien en español e inglés. Para otros idiomas considera modelos multilingües.
- **Tablas complejas**: PyPDFLoader extrae tablas como texto plano, lo que puede degradar la calidad en documentos muy tabulares.

---

## Próximos pasos sugeridos

Si quieres extender este proyecto:

- [ ] **Hybrid retrieval**: combinar búsqueda semántica con BM25 para mejor cobertura
- [ ] **Reranking**: añadir un modelo de reranking (ej. `BGE Reranker`) para mayor precisión
- [ ] **Evaluación**: implementar métricas RAG (faithfulness, answer relevancy) con RAGAS
- [ ] **Migrar a Qdrant**: para un entorno de producción con mayor rendimiento
- [ ] **Historial de conversación**: mantener contexto entre preguntas relacionadas
- [ ] **Soporte multimodal**: procesar imágenes y tablas en PDFs

---

## Recursos para aprender más

### Vídeos
- [Learn RAG From Scratch — Lance Martin (LangChain)](https://youtube.com/watch?v=sVcwVQRHIc8) — freeCodeCamp
- [Intro to Large Language Models — Andrej Karpathy](https://youtube.com/watch?v=zjkBMFhNj_g)

### Papers
- [RAG original paper — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)
- [Lost in the Middle — Liu et al., 2023](https://arxiv.org/abs/2307.03172)

### Documentación
- [LangChain docs](https://python.langchain.com/docs)
- [ChromaDB docs](https://docs.trychroma.com)
- [Chainlit docs](https://docs.chainlit.io)
- [Ollama models](https://ollama.com/library)

---
📄 [Descargar documento PDF](Arquitectura_RAG.pdf)

## Licencia

MIT — libre para usar, modificar y distribuir con atribución.

---

*Desarrollado como recurso educativo para la charla RAG — 2025*
