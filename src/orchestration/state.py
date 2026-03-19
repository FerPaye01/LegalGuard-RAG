"""
Estructura del Estado (State) para el LangGraph Orchestrator (Opción B: Estructurado).
Garantiza la trazabilidad para la UI de Confianza y evita desbordamientos de memoria.
"""
from typing import TypedDict, Annotated, List, Any
from langchain_core.messages import BaseMessage
import operator


class LegalGuardState(TypedDict):
    # El historial de chat. operator.add asegura que los mensajes se anexen, no se sobrescriban.
    messages: Annotated[List[BaseMessage], operator.add]
    
    # La pregunta actual optimizada para búsqueda vectorial
    optimized_query: str
    
    # Lista de diccionarios con los chunks recuperados en el turno ACTUAL (se sobrescribe para ahorrar RAM)
    context_docs: List[dict]
    
    # Mapeo de PII recuperado de Azure Search para que el Deanonymizer pueda revertir los hashes
    pii_mapping: List[dict]
    
    # Trazabilidad para la UI de Streamlit (ej. "Buscando en Azure...", "Ejecutando Python...")
    current_step: str
    
    # Bandera para el Circuit Breaker (Límite de reintentos)
    retry_count: int
