import os
import sys

# Añadir la raíz del proyecto al path
sys.path.append(os.getcwd())

from src.ingestion.pipeline import create_index_if_not_exists
from src.utils.logger import log_info

if __name__ == "__main__":
    log_info("Iniciando actualización forzada de esquema...")
    create_index_if_not_exists()
    log_info("Actualización de esquema solicitada a Azure.")
