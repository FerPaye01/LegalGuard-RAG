"""
Tool de validación financiera en entorno aislado (Dynamic Sessions).
Se inyecta un preámbulo inmutable para proteger el parseo monetario y forzar errors='raise' en Pandas.
"""
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from langchain_core.tools import tool

from src.config.settings import get_settings
from src.utils.logger import log_sequence, log_warn, log_info, log_error


# Preámbulo inmutable inyectado antes de cualquier script generado por el LLM.
# Contiene la lógica defensiva para evitar fallos de conversión de divisas.
_PREAMBULO_DATA_CLEANING = """
import pandas as pd
import re
import numpy as np

# Herramienta precargada para que el agente limpie strings monetarios antes de operarlos.
# Uso obligatorio antes de conversiones matemáticas para evitar que "errors='coerce'" genere NaNs silenciosos.
def limpiar_moneda_global(val):
    if pd.isna(val):
        return np.nan
    
    val_str = str(val)
    # Reemplaza todo lo que no sea dígito, punto o guión negativo
    num = re.sub(r'[^0-9\.\-]', '', val_str)
    
    try:
        if num == '': return np.nan
        return float(num)
    except ValueError:
         return np.nan
    
# Parche de seguridad: Sobrescribir pd.to_numeric en memoria para forzar la detención si falla la limpieza.
_original_to_numeric = pd.to_numeric
def _safe_to_numeric(*args, **kwargs):
    kwargs['errors'] = 'raise'
    return _original_to_numeric(*args, **kwargs)
pd.to_numeric = _safe_to_numeric
"""


def get_sessions_tool() -> SessionsPythonREPLTool:
    """Inicializa el cliente de ejecución sobre Azure Container Apps."""
    settings = get_settings()
    pool_endpoint = settings.azure_container_app_session_pool
    
    if not pool_endpoint:
        raise ValueError("AZURE_CONTAINER_APP_SESSION_POOL no está configurado.")
        
    return SessionsPythonREPLTool(pool_management_endpoint=pool_endpoint)


@tool
def ejecutar_analisis_financiero(codigo_agente: str) -> str:
    """
    Ejecuta el código pandas del Agente usando un pool aislado en Azure.
    Antes de ejecutar el código, esta herramienta inyecta la función `limpiar_moneda_global(valor)`.
    Úsala para analizar tablas extraídas de contratos, sumar multas o verificar balances.
    ATENCIÓN: pd.to_numeric() fallará (errors='raise') si no usas limpiar_moneda_global() antes en tus columnas de divisas.
    """
    log_sequence("dynamic-sessions: Preparando ejecución de script", f"{len(codigo_agente)} caracteres")
    
    try:
        repl = get_sessions_tool()
    except Exception as e:
        log_error("dynamic-sessions: Error inicializando el REPL Tool", e)
        return f"Error interno: {str(e)}"
    
    # Empaquetado del Sandbox con el Preámbulo
    codigo_completo = f"{_PREAMBULO_DATA_CLEANING}\n\n# --- COMIENZA EL CODIGO DEL AGENTE ---\n{codigo_agente}"
    
    log_info("dynamic-sessions: Ejecutando código en Azure Container Apps...")
    
    try:
        # Ejecución remota
        resultado = repl.execute(codigo_completo)
        log_info("dynamic-sessions: Ejecución completada")
        return str(resultado)
    except Exception as e:
        log_warn("dynamic-sessions: Fallo en la ejecución del script", e)
        return f"ValueError / Ejecución fallida: {str(e)}\n\nRecuerda usar limpiar_moneda_global() antes de hacer operaciones o to_numeric."
