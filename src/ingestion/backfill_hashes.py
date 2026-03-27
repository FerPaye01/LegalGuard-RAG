import os
import sys
import hashlib
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Añadir la raíz del proyecto al path
sys.path.append(os.getcwd())

from src.utils.logger import log_info, log_error, log_sequence

load_dotenv()

# Configuración
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "contratos-raw"
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "legalguard-index")

search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def backfill_hashes():
    log_sequence("Iniciando Backfill de Hashes SHA256", "Mantenimiento")
    
    if not container_client.exists():
        log_error(f"Fallo: El contenedor {CONTAINER_NAME} no existe.", None)
        return

    blobs = list(container_client.list_blobs())
    log_info(f"Encontrados {len(blobs)} archivos en el storage.")

    for blob in blobs:
        log_info(f"Procesando: {blob.name}")
        
        # 1. Obtener contenido del blob para calcular hash
        blob_data = container_client.download_blob(blob.name).readall()
        file_hash = compute_hash(blob_data)
        
        # 2. Buscar fragmentos en Azure Search con el mismo source_file
        # Azure Search permite actualizaciones parciales si enviamos la llave (id)
        results = search_client.search(
            search_text="*",
            filter=f"source_file eq '{blob.name}'",
            select=["id"]
        )
        
        updates = []
        for doc in results:
            updates.append({"id": doc["id"], "file_hash": file_hash})
        
        if updates:
            search_client.merge_documents(documents=updates)
            log_info(f"✅ Actualizados {len(updates)} fragmentos de {blob.name} con hash {file_hash[:10]}...")
        else:
            log_warn(f"⚠️ No se encontraron fragmentos para {blob.name} en el índice.")

    log_sequence("BACKFILL COMPLETADO", "Integridad SHA256 garantizada.")

if __name__ == "__main__":
    backfill_hashes()
