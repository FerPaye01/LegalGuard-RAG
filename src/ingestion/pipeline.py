import os
import time
import hashlib
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile
)
from src.ingestion.document_processor import extract_document_hybrid
from src.utils.logger import log_info, log_sequence, log_warn, log_error
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    log_warn("Falta instalar langchain-text-splitters. Haz un pip install langchain-text-splitters")

load_dotenv()

# Variables
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "contratos-raw"
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "legalguard-index")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") # Ej. text-embedding-3-small

# Configuración Clientes
oai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version=OPENAI_VERSION,
    azure_endpoint=OPENAI_ENDPOINT
)
search_index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))

def create_index_if_not_exists():
    log_sequence("Sincronizando esquema de índice en AI Search", INDEX_NAME)
    
    # Esquema del RAG: ID, ArchivoOrigen, Chunk de texto, Hash y Vector
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="source_file", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="file_hash", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]
    
    # Configuración de HNSW como indica el skill del proyecto
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )
    
    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    
    # create_or_update_index es más robusto para evoluciones de esquema (Hito Final)
    search_index_client.create_or_update_index(index)
    log_info(f"✅ Esquema del índice '{INDEX_NAME}' sincronizado.")

def compute_file_hash(file_bytes: bytes) -> str:
    """Genera la huella digital SHA256 del contenido binario de un archivo."""
    return hashlib.sha256(file_bytes).hexdigest()

def check_duplicate_by_hash(file_hash: str) -> dict:
    """
    Consulta Azure Search para saber si ya existe un documento con este hash exacto.
    Retorna: {'status': 'new' | 'duplicate' | 'new_version', 'existing_file': str | None}
    """
    try:
        results = list(search_client.search(
            search_text="*",
            filter=f"file_hash eq '{file_hash}'",
            select=["source_file", "file_hash"],
            top=1
        ))
        if results:
            existing_name = results[0].get("source_file", "")
            log_warn("Duplicado detectado", f"Hash existente en: {existing_name}")
            return {"status": "duplicate", "existing_file": existing_name}
        return {"status": "new", "existing_file": None}
    except Exception as e:
        log_error("Error consultando duplicados en Azure Search", e)
        return {"status": "new", "existing_file": None}

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def get_embedding_with_retry(text):
    response = oai_client.embeddings.create(input=text, model=EMBEDDING_DEPLOYMENT)
    return response.data[0].embedding

def smart_chunking(markdown_text):
    """
    Parámetro de Arquitecto: Ignoramos los "512 tokens estrictos".
    Partimos amistosamente por subtítulos de Markdown y, si queda muy gigantesco,
    lo rompemos cuidando de no quebrar código HTML de tablas.
    """
    # 1. Romper por jerarquía lógica de la Ley o Contrato
    headers_to_split_on = [
        ("#", "Título Principal"),
        ("##", "Capítulo"),
        ("###", "Artículo"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    
    # 2. Control de daños para cláusulas anormalmente largas o tablas gigantes (Límite suave 2000 chars)
    # Colocamos "</table>" como separador premium para evitar que un chunk se trague la mitad de una tabla.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, chunk_overlap=250,
        separators=["\n## ", "\n### ", "</table>\n", "</p>", "\n\n", "\n", " ", ""]
    )
    
    final_chunks = text_splitter.split_documents(md_header_splits)
    return [chunk.page_content for chunk in final_chunks]

def index_document_from_text(filename: str, markdown_text: str, file_hash: str = None):
    """
    Indexa un documento directamente desde su texto markdown (para la UI).
    Si se provee file_hash, lo adjunta a cada chunk para auditoría.
    """
    create_index_if_not_exists()
    
    chunks = smart_chunking(markdown_text)
    log_info(f"Indexando {filename}: particionado en {len(chunks)} trozos.")
    
    documents_to_upload = []
    for i, chunk_text in enumerate(chunks):
        vector = get_embedding_with_retry(chunk_text)
        safe_id = "".join(c for c in f"{filename}_{i}" if c.isalnum() or c in "-_")
        
        doc = {
            "id": safe_id,
            "source_file": filename,
            "content": chunk_text,
            "content_vector": vector
        }
        if file_hash:
            doc["file_hash"] = file_hash
        documents_to_upload.append(doc)
    
    if documents_to_upload:
        search_client.upload_documents(documents=documents_to_upload)
        log_info(f"✅ {len(documents_to_upload)} vectores inyectados de {filename}")
        return True
    return False

def process_pipeline():
    log_sequence("Arrancando el Pipeline Batch Secuencial (Opción A)", "ETL")
    
    if not CONNECTION_STRING:
        log_error("Falta AZURE_STORAGE_CONNECTION_STRING en el .env", None)
        return

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    if not container_client.exists():
        log_error(f"El contenedor '{CONTAINER_NAME}' de origen no existe.", None)
        return

    # Garatizar Base Vectorial viva
    create_index_if_not_exists()
    
    log_info(f"Escaneando contenedor '{CONTAINER_NAME}'...")
    blobs = list(container_client.list_blobs())
    
    temp_dir = "./data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    total_docs = 0
    for blob in blobs:
        if blob.name.lower().endswith(".pdf"):
            log_sequence(f"Procesando Blob", blob.name)
            temp_path = os.path.join(temp_dir, blob.name)
            with open(temp_path, "wb") as f:
                f.write(container_client.download_blob(blob.name).readall())
                
            md_content = extract_document_hybrid(temp_path)
            if md_content:
                if index_document_from_text(blob.name, md_content):
                    total_docs += 1
            os.remove(temp_path)
            
    log_sequence("PIPELINE 100% COMPLETADO", f"{total_docs} documentos PDF vectorizados.")


if __name__ == "__main__":
    process_pipeline()
