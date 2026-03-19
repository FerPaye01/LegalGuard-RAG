# 🚀 Instrucciones de Pruebas y Claves (LegalGuard MVP)

Para probar todo el flujo que hemos programado (Ingesta + LangGraph), es indispensable contar con las APIs de los servicios en Azure.

## 1. Archivo de Secretos (`.env`)
Debes crear un archivo llamado exactamente `.env` en la raíz del proyecto (`e:\OSCAR\HACKATONES\azure-1\.env`).

Copia y pega este contenido literal y rellena los valores:

```env
# ==== Azure OpenAI (Generación y Embeddings) ====
AZURE_OPENAI_API_KEY="tu_clave_aqui"
AZURE_OPENAI_ENDPOINT="https://tu_recurso.openai.azure.com/"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# ==== Azure AI Search (Búsqueda Vectorial) ====
AZURE_SEARCH_API_KEY="tu_clave_aqui"
AZURE_SEARCH_ENDPOINT="https://tu_recurso.search.windows.net"
AZURE_SEARCH_INDEX_NAME="contratos-index"

# ==== Azure Document Intelligence (Extracción de PDF) ====
AZURE_FORM_RECOGNIZER_KEY="tu_clave_aqui"
AZURE_FORM_RECOGNIZER_ENDPOINT="https://tu_recurso.cognitiveservices.azure.com/"

# ==== Microsoft Presidio (Seguridad y Anonimización) ====
PRESIDIO_ENCRYPTION_KEY="ClaveDe32CaracteresUltraSecreta!"

# ==== Azure Container Apps (Intérprete Matemático Seguro) ====
AZURE_CONTAINER_APP_SESSION_POOL="https://tu_pool.xxx.azurecontainerapps.io"
```

## 2. Cómo y Dónde Conseguir las Apis (Portal de Azure)

- **Azure OpenAI**: Busca el recurso "Azure OpenAI" en tu portal. Ve a "Keys and Endpoints". Copia la *Key 1* y el *Endpoint*. Adentro en "Model Deployments" busca o crea tus modelos (`gpt-4o` y `text-embedding-ada-002`).
- **Azure AI Search**: Busca el recurso "Search services". Ve a "Keys". Copia la "Primary admin key" y el "Url".
- **Azure Document Intelligence**: Busca "Document Intelligence". Ve a "Keys and Endpoint". Copia la *Key 1* y el *Endpoint*.
- **Presidio Encryption Key**: Simplemente inventa una clave aleatoria tuya de 32 o 44 caracteres alfanuméricos. Es el candado ("sal") AES para encriptar los DNIS antes de subirlos a la nube.
- **Dynamic Sessions Pool**: Se obtiene al crear un _Session Pool_ desde Azure Container Apps. (Ve a la documentación de Azure Container Apps Dynamic Sessions y copia el Management Endpoint).

## 3. Pruebas Rápidas Paso a Paso

Una vez que tengas el `.env` lleno y el entorno (`.venv`) activado:

### Prueba de Ingesta
Crea un archivo rápido en Python (ej. `test_ingesta.py`) en la raíz con esto:
```python
import os
from dotenv import load_dotenv
from src.ingestion.document_processor import process_pdf
from src.retrieval.azure_search_client import create_or_update_index, upload_documents

# Cargar del .env
load_dotenv()

# Ejecutar
create_or_update_index()
documentos = process_pdf("ruta/a/un_contrato_de_prueba.pdf")
upload_documents(documentos)
print("¡Ingesta Completa!")
```

### Prueba de Orquestación (Cerebro)
Crea un `test_grafo.py` en la raíz con esto:
```python
from dotenv import load_dotenv
from src.orchestration.graph import build_graph
from langchain_core.messages import HumanMessage

load_dotenv()

app = build_graph()
config = {"configurable": {"thread_id": "prueba_1"}, "recursion_limit": 5}

inputs = {"messages": [HumanMessage(content="¿Cuál es la multa del contrato de la empresa anonimizada?")]}

for output in app.stream(inputs, config=config):
    # Ver paso actual ejecutándose "pensamiento"
    for key, value in output.items():
        if "current_step" in value:
            print(f"➜ Agente pensando: {value['current_step']}")
        
        if "messages" in value and value["messages"]:
            print(f"🤖 Respuesta Final: {value['messages'][-1].content}")
```
