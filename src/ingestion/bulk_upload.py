import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Importamos nuestro sistema central de logs tal como mandan las directivas
from src.utils.logger import log_info, log_sequence, log_error

load_dotenv()

def upload_folder_to_blob():
    """
    Sube masivamente todos los PDFs de una carpeta local hacia un contenedor 
    específico en Azure Blob Storage.
    """
    log_sequence("Iniciando subida masiva a Azure Storage", "BlobService")
    
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "contratos-raw"  # Cambia por tu contenedor real si es diferente
    source_folder = "./data/pdfs"     # Cambia a tu carpeta de PDFs locales real (ej. ./contratos_cuad)
    
    if not connection_string:
        log_error("Falta la variable AZURE_STORAGE_CONNECTION_STRING en el .env", None)
        return

    if not os.path.exists(source_folder):
        log_error(f"La carpeta origen '{source_folder}' no existe. Por favor, créala.", None)
        return

    try:
        service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = service_client.get_container_client(container_name)
        
        # Opcional: Intentar crear el contenedor si no existe previamente
        if not container_client.exists():
            container_client.create_container()
            log_info(f"Contenedor '{container_name}' creado en Storage.")

        pdf_count = 0
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(source_folder, filename)
                blob_client = container_client.get_blob_client(filename)
                
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                    log_info(f"✅ Subido al Storage: {filename}")
                    pdf_count += 1
                    
        log_sequence("Proceso de Bulk Upload finalizado", f"Total enviados: {pdf_count} PDFs")
        
    except Exception as e:
        log_error("Ocurrió un error en la conexión o subida de Blob Storage", e)


if __name__ == "__main__":
    print("🚀 INICIANDO HERRAMIENTA DE INGESTA MASIVA RAG\n")
    upload_folder_to_blob()
