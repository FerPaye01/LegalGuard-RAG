"""
Configuración centralizada de variables de entorno (Pydantic Settings)
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str = "contratos-index"

    # Azure Blob Storage
    azure_storage_connection_string: str
    azure_storage_container: str = "contratos"

    # Azure Document Intelligence
    azure_form_recognizer_endpoint: str
    azure_form_recognizer_key: str

    # Azure Content Safety
    azure_content_safety_endpoint: str
    azure_content_safety_key: str

    # Presidio — Clave AES (128, 192 o 256 bits)
    presidio_encryption_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    # Retorna instancia única de configuración (cacheada)
    return Settings()
