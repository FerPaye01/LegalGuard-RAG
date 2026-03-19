"""
Cliente de Azure AI Search: Creación de índice, configuración de HNSW/Semantic Ranker y subida de lotes.
"""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from langchain_openai import AzureOpenAIEmbeddings

from src.config.settings import get_settings
from src.utils.logger import log_info, log_error, log_sequence, log_debug


def _get_index_client() -> SearchIndexClient:
    settings = get_settings()
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_api_key)
    )


def _get_search_client() -> SearchClient:
    settings = get_settings()
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(settings.azure_search_api_key)
    )


def _get_embeddings_client() -> AzureOpenAIEmbeddings:
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=settings.azure_openai_api_version
    )


def create_or_update_index():
    """Crea el índice con el esquema Ultra-Auditable (Opción B), HNSW y Semantic Ranker."""
    settings = get_settings()
    index_client = _get_index_client()
    index_name = settings.azure_search_index_name

    log_sequence("azure-search: Verificando/Creando índice", index_name)

    # 1. Definición de Campos (Schema Opción B)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="document_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="es.microsoft"),
        
        # Campo Vectorial (1536 dimensiones para text-embedding-ada-002)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw_profile"
        ),
        
        # Metadatos de Auditoría y MLOps
        SearchableField(name="source_file", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SimpleField(name="has_table", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="is_degraded", type=SearchFieldDataType.Boolean, filterable=True),
        
        # Guardamos el JSON string de las entidades PII
        SimpleField(name="presidio_entities", type=SearchFieldDataType.String, filterable=False),
    ]

    # 2. Configuración de Búsqueda Vectorial (HNSW)
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw_config")],
        profiles=[VectorSearchProfile(name="hnsw_profile", algorithm_configuration_name="hnsw_config")]
    )

    # 3. Configuración del Semantic Ranker (Crítico para la defensa contra ceguera de atención)
    semantic_config = SemanticConfiguration(
        name="semantic_config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="source_file"),
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="document_type")]
        )
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Ensamblar y crear índice
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    try:
        index_client.create_or_update_index(index)
        log_info(f"azure-search: Índice '{index_name}' aprovisionado correctamente con HNSW y Semantic Ranker.")
    except Exception as e:
        log_error(f"azure-search: Error al crear el índice: {e}")
        raise


def upload_documents(documents: list[dict]):
    """Genera embeddings para el contenido y sube el lote a Azure AI Search."""
    if not documents:
        log_warn("azure-search: No hay documentos para subir.")
        return

    log_sequence("azure-search: Preparando subida de lote", f"{len(documents)} chunks")
    
    # Generar vectores usando LangChain AzureOpenAIEmbeddings
    embedder = _get_embeddings_client()
    texts_to_embed = [doc["content"] for doc in documents]
    
    log_debug("azure-search: Calculando embeddings (ada-002)...", "")
    try:
        embeddings = embedder.embed_documents(texts_to_embed)
    except Exception as e:
        log_error(f"azure-search: Falló la generación de embeddings: {e}")
        return

    # Inyectar el vector en cada documento
    for doc, vector in zip(documents, embeddings):
        doc["content_vector"] = vector

    search_client = _get_search_client()
    
    # Subida en batch
    try:
        result = search_client.upload_documents(documents=documents)
        success_count = sum(1 for r in result if r.succeeded)
        log_info(f"azure-search: Se subieron {success_count}/{len(documents)} documentos exitosamente.")
        
        if success_count < len(documents):
            log_warn("azure-search: Algunos documentos fallaron durante la ingesta. Revisa los logs de Azure.")
            
    except Exception as e:
        log_error(f"azure-search: Excepción crítica al subir documentos: {e}")
