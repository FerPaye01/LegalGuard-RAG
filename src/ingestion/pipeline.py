import os
import time
import hashlib
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
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
        # Metadatos enriquecidos (Document Selector Pro)
        SimpleField(name="upload_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name="doc_summary", type=SearchFieldDataType.String),
        SearchableField(name="doc_entities", type=SearchFieldDataType.String),
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

# Prompts diferenciados para el Onboarding por Rol (Persona Dinámica)
PERSONA_PROMPTS = {
    "Legal": "Identifica los 3 mayores riesgos legales (terminación, indemnización, jurisdicción). Sé directo y técnico.",
    "Financiero": "Identifica los términos de pago, montos exactos, penalidades financieras y fechas críticas de facturación.",
    "Salud": "Identifica los protocolos de cumplimiento, normativas sanitarias, obligaciones de bioseguridad y plazos de reporte.",
    "Orchestrator": "Resume los puntos clave del documento de forma concisa y equilibrada."
}

def generate_doc_metadata(markdown_text: str, persona: str = "Orchestrator") -> dict:
    """Genera resumen contextualizado por rol y entidades usando GPT-4o en la ingesta."""
    try:
        context = markdown_text[:6000]
        instruccion_rol = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Orchestrator"])
        
        response = oai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{
                "role": "user",
                "content": f"""Eres un analista experto. Analiza este documento para un profesional con rol: {persona}.
{instruccion_rol}
Responde SOLO en JSON con este formato exacto:
{{"summary": "resumen de 2-3 líneas orientado al rol '{persona}'", "entities": "lista de entidades clave separadas por comas: partes involucradas, montos, fechas relevantes"}}

Texto del contrato:
{context}"""
            }],
            response_format={"type": "json_object"},
            max_tokens=350,
            temperature=0.1
        )
        import json
        result = json.loads(response.choices[0].message.content)
        return {
            "doc_summary": result.get("summary", ""),
            "doc_entities": result.get("entities", "")
        }
    except Exception as e:
        log_warn("No se pudo generar metadata enriquecida", str(e))
        return {"doc_summary": "", "doc_entities": ""}

def get_blob_sas_url(filename: str, expiry_hours: int = 1) -> str:
    """Genera una URL firmada (SAS) para abrir el PDF en nueva pestaña."""
    try:
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            return ""
        
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        account_name = blob_service.account_name
        account_key = blob_service.credential.account_key
        
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=CONTAINER_NAME,
            blob_name=filename,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
        )
        return f"https://{account_name}.blob.core.windows.net/{CONTAINER_NAME}/{filename}?{sas_token}"
    except Exception as e:
        log_error("Error generando SAS URL", e)
        return ""

def get_available_documents_enriched() -> list:
    """Devuelve lista completa con metadatos enriquecidos para el Document Selector Pro.
    Usa fallback: primero intenta campos enriquecidos, si falla usa solo source_file.
    """
    # Fase 1: Intentar con SELECT completo (campos nuevos del esquema)
    try:
        results = search_client.search(
            search_text="*",
            select=["source_file", "upload_date", "doc_summary", "doc_entities"],
            top=1000
        )
        seen = {}
        for doc in results:
            fname = doc.get("source_file", "")
            if fname and fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "upload_date": doc.get("upload_date", ""),
                    "summary": doc.get("doc_summary", ""),
                    "entities": doc.get("doc_entities", "")
                }
        if seen:
            return sorted(seen.values(), key=lambda x: x["upload_date"], reverse=True)
    except Exception as e:
        log_warn(f"Select enriquecido falló ({e}), usando fallback básico")

    # Fase 2 (Fallback): Solo source_file — siempre funciona aunque el esquema sea viejo
    try:
        results = search_client.search(search_text="*", select=["source_file"], top=1000)
        seen = {}
        for doc in results:
            fname = doc.get("source_file", "")
            if fname and fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "upload_date": "",
                    "summary": "Metadatos no disponibles (re-ingesta recomendada).",
                    "entities": ""
                }
        return sorted(seen.values(), key=lambda x: x["filename"])
    except Exception as e:
        log_error("Error en fallback básico de documentos", e)
        return []

def get_available_documents():
    """Devuelve la lista única de archivos cargados en el índice de Azure sin cargar el motor de IA."""
    try:
        results = search_client.search(search_text="*", select=["source_file"], top=1000)
        unique_files = sorted(list(set(doc["source_file"] for doc in results if doc.get("source_file"))))
        return unique_files
    except Exception as e:
        log_error("No se pudo recuperar la lista de documentos de Azure", e)
        return []

def index_document_from_text(filename: str, markdown_text: str, file_hash: str = None):
    """
    Indexa un documento desde su texto markdown.
    Genera metadatos enriquecidos (resumen, entidades) via LLM durante la ingesta.
    """
    create_index_if_not_exists()
    
    # Generación de metadatos en la ingesta (una sola llamada LLM por documento)
    log_sequence("Generando metadatos enriquecidos", filename)
    metadata = generate_doc_metadata(markdown_text)
    upload_date = datetime.now(timezone.utc).isoformat()
    
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
            "content_vector": vector,
            "upload_date": upload_date,
            "doc_summary": metadata["doc_summary"],
            "doc_entities": metadata["doc_entities"]
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
