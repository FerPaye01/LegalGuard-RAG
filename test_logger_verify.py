from src.utils.logger import log_info, log_sequence, log_error, log_warn
import os

def test():
    print("--- Probando Logger ---")
    log_info("Sistema", "Iniciando prueba de verificación")
    log_sequence("RAG", "Recuperando documentos de prueba")
    log_warn("Config", "Detectada configuración por defecto")
    log_error("API", "Error simulado de conexión")
    
    log_file = "outputs/logs/legalguard.log"
    if os.path.exists(log_file):
        print(f"\n✅ Archivo de log creado en: {log_file}")
    else:
        print("\n❌ Error: El archivo de log no fue creado.")

if __name__ == "__main__":
    test()
