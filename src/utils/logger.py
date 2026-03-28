import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Códigos de color ANSI para la terminal
class LogColors:
    RESET = "\033[0m"
    DEBUG = "\033[36m"    # Cyan
    INFO = "\033[32m"     # Verde
    WARNING = "\033[33m"  # Amarillo
    ERROR = "\033[31m"    # Rojo
    CRITICAL = "\033[41m" # Fondo Rojo

class ColorFormatter(logging.Formatter):
    """Formateador personalizado para inyectar colores en la consola."""
    FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

    def format(self, record):
        log_color = getattr(LogColors, record.levelname, LogColors.RESET)
        format_str = f"{log_color}{self.FORMAT}{LogColors.RESET}"
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Configuración del Logger Principal
def _setup_logger():
    logger = logging.getLogger("LegalGuard")
    logger.setLevel(logging.DEBUG) # Nivel base

    # 1. Handler para la Consola (Con colores)
    # Azure App Service y Container Apps capturan stdout automáticamente
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())

    # 2. Handler para Azure Monitor (Application Insights)
    # Reemplaza la necesidad de archivos locales persistentes
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") or os.getenv("AZURE_MONITOR_CONNECTION_STRING")
    if connection_string:
        try:
            from opencensus.ext.azure.log_exporter import AzureLogHandler
            az_handler = AzureLogHandler(connection_string=connection_string)
            az_handler.setLevel(logging.INFO)
            logger.addHandler(az_handler)
        except Exception as e:
            print(f"⚠️ Error al configurar Azure Monitor: {e}")

    # Evitar duplicados
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)
        
    return logger

# Instancia global
_logger = _setup_logger()

# Funciones de utilidad exportadas
def log_debug(context: str, message: str = ""):
    _logger.debug(f"[{context}] {message}".strip())

def log_info(context: str, message: str = ""):
    _logger.info(f"[{context}] {message}".strip())

def log_warn(context: str, message: str = ""):
    _logger.warning(f"[{context}] {message}".strip())

def log_error(context: str, message: str = ""):
    _logger.error(f"[{context}] {message}".strip())

def log_sequence(context: str, message: str = ""):
    _logger.info(f"🚀 [{context}] >>> {message}".strip())

def log_critical(context: str, message: str = ""):
    _logger.critical(f"[{context}] {message}".strip())
