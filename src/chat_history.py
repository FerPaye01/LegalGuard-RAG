import os
import json
from datetime import datetime
from dotenv import load_dotenv
from src.utils.logger import log_info, log_sequence, log_warn, log_error, log_debug

load_dotenv()

# Singleton del cliente Cosmos
_cosmos_client = None
_container = None


def _get_container():
    """Inicializa el cliente de Cosmos DB y devuelve el contenedor de historial."""
    global _cosmos_client, _container
    
    if _container is not None:
        return _container
    
    conn_str = os.getenv("COSMOS_CONNECTION_STRING")
    if not conn_str:
        log_warn("COSMOS_CONNECTION_STRING no configurado. Historial volátil.")
        return None
    
    try:
        from azure.cosmos import CosmosClient, PartitionKey
        
        db_name = os.getenv("COSMOS_DATABASE_NAME", "LegalGuardDB")
        container_name = os.getenv("COSMOS_CONTAINER_NAME", "ChatHistory")
        
        _cosmos_client = CosmosClient.from_connection_string(conn_str)
        
        # Crear DB y contenedor si no existen (idempotente)
        database = _cosmos_client.create_database_if_not_exists(id=db_name)
        _container = database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/session_id"),
            offer_throughput=400
        )
        
        log_info(f"🟢 Cosmos DB conectado: {db_name}/{container_name}")
        return _container
    except Exception as e:
        log_error("Error conectando a Cosmos DB", e)
        return None


def save_chat_session(session_id: str, messages: list, persona: str = "Orchestrator"):
    """Guarda o actualiza la sesión de chat en Cosmos DB."""
    container = _get_container()
    if not container:
        return False
    
    # Limpiar mensajes para serialización (quitar objetos no serializables)
    clean_messages = []
    for msg in messages:
        clean_msg = {
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        }
        if "documents" in msg:
            clean_msg["documents"] = [
                {"source_file": d.get("source_file", ""), "content": d.get("content", "")[:500]}
                for d in msg["documents"]
            ]
        if "audit_score" in msg:
            clean_msg["audit_score"] = msg["audit_score"]
        clean_messages.append(clean_msg)
    
    doc = {
        "id": session_id,
        "session_id": session_id,
        "messages": clean_messages,
        "persona": persona,
        "message_count": len(clean_messages),
        "updated_at": datetime.utcnow().isoformat(),
        "created_at": datetime.utcnow().isoformat()
    }
    
    try:
        container.upsert_item(body=doc)
        log_info(f"💾 Sesión guardada en Cosmos: {session_id} ({len(clean_messages)} msgs)")
        return True
    except Exception as e:
        log_error("Error guardando sesión en Cosmos", e)
        return False


def load_chat_session(session_id: str) -> dict:
    """Carga una sesión de chat específica desde Cosmos DB."""
    container = _get_container()
    if not container:
        return None
    
    try:
        item = container.read_item(item=session_id, partition_key=session_id)
        log_info(f"📂 Sesión cargada desde Cosmos: {session_id}")
        return item
    except Exception:
        return None


def list_chat_sessions(max_sessions: int = 10) -> list:
    """Lista las sesiones de chat más recientes."""
    container = _get_container()
    if not container:
        return []
    
    try:
        query = "SELECT c.id, c.session_id, c.persona, c.message_count, c.updated_at FROM c ORDER BY c.updated_at DESC OFFSET 0 LIMIT @max"
        items = list(container.query_items(
            query=query,
            parameters=[{"name": "@max", "value": max_sessions}],
            enable_cross_partition_query=True
        ))
        log_debug("Cosmos", f"{len(items)} sesiones encontradas")
        return items
    except Exception as e:
        log_error("Error listando sesiones de Cosmos", e)
        return []


def delete_chat_session(session_id: str) -> bool:
    """Elimina una sesión de chat de Cosmos DB."""
    container = _get_container()
    if not container:
        return False
    
    try:
        container.delete_item(item=session_id, partition_key=session_id)
        log_info(f"🗑️ Sesión eliminada de Cosmos: {session_id}")
        return True
    except Exception as e:
        log_error("Error eliminando sesión de Cosmos", e)
        return False
