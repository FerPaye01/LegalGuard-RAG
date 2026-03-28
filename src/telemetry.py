import os
import time
from datetime import datetime
from dotenv import load_dotenv
from src.utils.logger import log_info, log_sequence, log_warn, log_error

load_dotenv()

# --- CAPA 1: Cronómetro de Nodos (Telemetría Interna) ---

class NodeTimer:
    """Cronómetro para medir la latencia de cada nodo del grafo LangGraph."""
    
    def __init__(self):
        self.timings = {}
        self._start_times = {}
    
    def start(self, node_name: str):
        """Marca el inicio de un nodo."""
        self._start_times[node_name] = time.perf_counter()
    
    def stop(self, node_name: str):
        """Marca el fin de un nodo y registra la duración."""
        if node_name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[node_name]
            self.timings[node_name] = round(elapsed * 1000)  # Milisegundos
            log_info(f"⏱️ Nodo [{node_name}]: {self.timings[node_name]}ms")
            del self._start_times[node_name]
    
    def get_report(self) -> dict:
        """Devuelve el reporte de latencia por nodo."""
        total = sum(self.timings.values())
        return {
            "nodes": self.timings.copy(),
            "total_ms": total,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def reset(self):
        """Reinicia el cronómetro."""
        self.timings = {}
        self._start_times = {}


# --- CAPA 2: Heartbeat de Servicios Azure ---

def check_azure_health() -> dict:
    """Verifica la salud de los 3 servicios core de Azure."""
    health = {}
    
    # 1. Azure OpenAI
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        # Llamada mínima para validar conectividad
        client.models.list()
        health["Azure OpenAI"] = {"status": "healthy", "icon": "🟢"}
    except Exception as e:
        health["Azure OpenAI"] = {"status": f"error: {str(e)[:60]}", "icon": "🔴"}
    
    # 2. Azure AI Search
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "contratos-index"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )
        # Buscar 1 doc para verificar conectividad
        list(search_client.search(search_text="*", top=1))
        health["Azure AI Search"] = {"status": "healthy", "icon": "🟢"}
    except Exception as e:
        health["Azure AI Search"] = {"status": f"error: {str(e)[:60]}", "icon": "🔴"}
    
    # 3. Azure Blob Storage
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if conn_str:
            blob_client = BlobServiceClient.from_connection_string(conn_str)
            blob_client.get_account_information()
            health["Azure Storage"] = {"status": "healthy", "icon": "🟢"}
        else:
            health["Azure Storage"] = {"status": "no config", "icon": "🟡"}
    except Exception as e:
        health["Azure Storage"] = {"status": f"error: {str(e)[:60]}", "icon": "🔴"}
    
    # 4. Content Safety
    try:
        from azure.ai.contentsafety import ContentSafetyClient
        from azure.core.credentials import AzureKeyCredential
        cs_endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        cs_key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        if cs_endpoint and cs_key:
            cs_client = ContentSafetyClient(cs_endpoint, AzureKeyCredential(cs_key))
            health["Content Safety"] = {"status": "healthy", "icon": "🟢"}
        else:
            health["Content Safety"] = {"status": "no config", "icon": "🟡"}
    except Exception as e:
        health["Content Safety"] = {"status": f"error: {str(e)[:60]}", "icon": "🔴"}
    
    return health


# --- CAPA 3: Application Insights (Exportador OpenCensus) ---

_insights_exporter = None

def init_application_insights():
    """Inicializa el exportador de Application Insights si la clave está configurada."""
    global _insights_exporter
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    if not connection_string:
        log_warn("Application Insights no configurado (falta APPLICATIONINSIGHTS_CONNECTION_STRING)")
        return False
    
    try:
        from opencensus.ext.azure.trace_exporter import AzureExporter
        from opencensus.trace.tracer import Tracer
        from opencensus.trace.samplers import AlwaysOnSampler
        
        _insights_exporter = AzureExporter(connection_string=connection_string)
        log_info("🔵 Application Insights conectado correctamente.")
        return True
    except Exception as e:
        log_error("Error conectando Application Insights", e)
        return False

def track_event(name: str, properties: dict = None):
    """Envía un evento personalizado a Application Insights."""
    if not _insights_exporter:
        return
    
    try:
        from opencensus.ext.azure.trace_exporter import AzureExporter
        from opencensus.trace.tracer import Tracer
        from opencensus.trace.samplers import AlwaysOnSampler
        
        tracer = Tracer(exporter=_insights_exporter, sampler=AlwaysOnSampler())
        with tracer.span(name=name) as span:
            if properties:
                for key, value in properties.items():
                    span.add_attribute(key, str(value))
        log_info(f"📡 Evento enviado a App Insights: {name}")
    except Exception as e:
        log_warn(f"No se pudo enviar evento a Insights: {e}")

def track_node_latency(node_timings: dict):
    """Envía las latencias del grafo como evento a Application Insights."""
    track_event("LegalGuard.GraphExecution", node_timings.get("nodes", {}))
