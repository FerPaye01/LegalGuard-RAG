"""
Pipeline de ingesta: PDF → Document Intelligence → Presidio → Semantic Chunking
"""
import json
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.config.settings import get_settings
from src.privacy.presidio_engine import anonymize_text
from src.utils.logger import log_debug, log_info, log_sequence, log_warn, log_error


def _get_doc_intelligence_client() -> DocumentIntelligenceClient:
    settings = get_settings()
    return DocumentIntelligenceClient(
        endpoint=settings.azure_form_recognizer_endpoint,
        credential=AzureKeyCredential(settings.azure_form_recognizer_key)
    )


def _get_embeddings() -> AzureOpenAIEmbeddings:
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=settings.azure_openai_api_version
    )


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(HttpResponseError)
)
def extract_layout_from_pdf(file_path: str) -> tuple[str, bool]:
    log_sequence("document-intelligence: Solicitando prebuilt-layout", file_path)

    client = _get_doc_intelligence_client()

    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body={"base64Source": __import__("base64").b64encode(pdf_bytes).decode()},
        output_content_format=ContentFormat.MARKDOWN
    )
    result = poller.result()

    content = result.content or ""
    has_table = "<table>" in content.lower()

    log_debug("document-intelligence: Layout extraído", f"Tablas detectadas: {has_table}, {len(content)} caracteres")

    return content, has_table


def semantic_chunk(text: str) -> tuple[list[str], bool]:
    log_sequence("semantic-chunking: Fragmentando documento", f"{len(text)} caracteres")
    embeddings = _get_embeddings()
    
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )

    try:
        chunks = chunker.split_text(text)
        is_degraded = False
        log_info(f"semantic-chunking: {len(chunks)} fragmentos semánticos generados")
    except Exception as e:
        log_error(f"semantic-chunking: Falló el chunker semántico. Aplicando Fallback. Error: {e}")
        fallback_chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = fallback_chunker.split_text(text)
        is_degraded = True
        log_info(f"semantic-chunking: {len(chunks)} fragmentos de fallback generados")

    for i, chunk in enumerate(chunks):
        token_estimate = len(chunk.split())
        if token_estimate > 500:
            log_warn(f"semantic-chunking: Fragmento {i} excede 500 tokens (~{token_estimate}), considera dividirlo")

    return chunks, is_degraded


def process_pdf(file_path: str, document_type: str = "contract") -> list[dict]:
    log_sequence("ingestion: Iniciando pipeline para", file_path)

    raw_text, has_table = extract_layout_from_pdf(file_path)

    if not raw_text:
        log_error("ingestion: Document Intelligence no devolvió contenido", file_path)
        return []

    # FIX 1: Preservar metadatos de entidades anonimizadas
    clean_text, anonymized_items = anonymize_text(raw_text)
    
    entities_list = [
        {"entity_type": item.entity_type, "start": item.start, "end": item.end} 
        for item in anonymized_items
    ]
    presidio_entities_str = json.dumps(entities_list)

    # FIX 3: Manejo de chunks e is_degraded
    chunks, is_degraded = semantic_chunk(clean_text)

    if not chunks:
        log_warn("ingestion: No se generaron fragmentos para el documento", file_path)
        return []

    document_id = file_path.split("/")[-1].replace(".pdf", "").replace("\\", "_")
    documents = [
        {
            "id": f"{document_id}_chunk_{i}",
            "content": chunk,
            "document_id": document_id,
            "document_type": document_type,
            "chunk_index": i,
            "has_table": has_table,
            "is_degraded": is_degraded,
            "presidio_entities": presidio_entities_str,
            "source_file": file_path,
        }
        for i, chunk in enumerate(chunks)
    ]

    log_info(f"ingestion: Pipeline completado. {len(documents)} documentos listos para indexar")
    return documents
