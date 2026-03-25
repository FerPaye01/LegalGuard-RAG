import logging
import sys
from pathlib import Path

# Crear el directorio de logs si no existe
LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "legalguard.log"

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
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())

    # 2. Handler para el Archivo (Sin colores, detallado para auditoría)
    file_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)-8s | %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Evitar duplicados si se llama múltiples veces
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

# Instancia global
_logger = _setup_logger()

# Funciones de utilidad exportadas para el resto de módulos
def log_debug(context: str, message: str = ""):
    _logger.debug(f"[{context}] {message}".strip())

def log_info(context: str, message: str = ""):
    _logger.info(f"[{context}] {message}".strip())

def log_warn(context: str, message: str = ""):
    _logger.warning(f"[{context}] {message}".strip())

def log_error(context: str, message: str = ""):
    _logger.error(f"[{context}] {message}".strip())

def log_sequence(context: str, message: str = ""):
    """Útil para marcar el inicio de pasos importantes en LangGraph o Ingesta."""
    _logger.info(f"🚀 [{context}] >>> {message}".strip())

def log_critical(context: str, message: str = ""):
    _logger.critical(f"[{context}] {message}".strip())
