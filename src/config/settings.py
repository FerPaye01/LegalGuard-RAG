from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_chat_deployment: str = "gpt-4o"  # Añadido para hacer match con tu .env
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"

    # Azure AI Search
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str = "contratos-index"

    # Azure Container Apps (Intérprete de Código)
    azure_container_app_session_pool: str = "" # Añadido para hacer match con tu .env

    # Azure Blob Storage (Opcionales por ahora para que no crashee)
    azure_storage_connection_string: str = "dummy_string"
    azure_storage_container: str = "contratos"

    # Azure Document Intelligence
    azure_form_recognizer_endpoint: str
    azure_form_recognizer_key: str

    # Azure Content Safety (Opcionales por ahora)
    azure_content_safety_endpoint: str = "https://dummy.cognitiveservices.azure.com/"
    azure_content_safety_key: str = "dummy_key"

    # Presidio — Clave AES
    presidio_encryption_key: str

    # Configuración de Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"  # ¡ESTO EVITA EL ERROR DE 'extra_forbidden'!
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
