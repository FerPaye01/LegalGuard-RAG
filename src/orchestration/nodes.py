"""
Definición de Nodos (Nodes) para el flujo de LangGraph.
Contiene la lógica encapsulada de cada paso del pipeline de LegalGuard RAG.
"""
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_transformers import LongContextReorder

from src.config.settings import get_settings
from src.retrieval.azure_search_client import _get_search_client
from src.orchestration.state import LegalGuardState
from src.privacy.presidio_engine import deanonymize_text
from src.utils.logger import log_sequence, log_warn, log_error, log_info


def _get_llm() -> AzureChatOpenAI:
    settings = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_chat_deployment,
        api_version=settings.azure_openai_api_version,
        temperature=0.0
    )


def rewrite_query(state: LegalGuardState):
    """Nodo: Reescribe la pregunta original para maximizar la búsqueda vectorial."""
    log_sequence("nodo: rewrite_query", "Generando query optimizada")
    
    last_message = state["messages"][-1].content
    llm = _get_llm()
    
    prompt = f"Reescribe esta pregunta para búsqueda vectorial en contratos legales. Extrae solo las palabras clave importantes: '{last_message}'"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "optimized_query": response.content,
        "current_step": "Optimizando pregunta con LLM"
    }


def retrieve_contracts(state: LegalGuardState):
    """Nodo: Recupera de Azure AI Search (HNSW + Semantic Ranker) y reordena."""
    query = state.get("optimized_query", state["messages"][-1].content)
    log_sequence("nodo: retrieve_contracts", f"Buscando: {query}")
    
    search_client = _get_search_client()
    
    try:
        # Recuperación Hibrida + Semántica con Top 10
        results = search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="semantic_config",
            top=10
        )
        
        raw_docs = []
        pii_mappings = []
        for result in results:
            if result.get("@search.rerankerScore", 0) > 1.5:  # Poda de falsos positivos
                raw_docs.append({
                    "id": result["id"],
                    "content": result["content"],
                    "score": result.get("@search.rerankerScore")
                })
                
                # Extraer entidades PII inyectadas durante la ingesta
                if result.get("presidio_entities"):
                    try:
                        entities = json.loads(result["presidio_entities"])
                        pii_mappings.extend(entities)
                    except Exception:
                        pass
        
        # Skill: LangChain LongContextReorder (Márgenes de visión)
        if len(raw_docs) > 3:
            log_info(f"Aplicando LongContextReorder a {len(raw_docs)} documentos")
            reordering = LongContextReorder()
            # transform_documents espera objetos tipo 'Document', así que simulamos o pasamos el texto. 
            # Como LangChain requiere Document.page_content, mapearemos internamente.
            from langchain_core.documents import Document
            lc_docs = [Document(page_content=d["content"], metadata={"id": d["id"]}) for d in raw_docs]
            reordered = reordering.transform_documents(lc_docs)
            
            # Devolvemos el string limpio reordenado
            final_docs = [{"content": d.page_content} for d in reordered]
        else:
            final_docs = raw_docs
            
    except Exception as e:
        log_error("Fallo la búsqueda vectorial", e)
        final_docs = []
        pii_mappings = []
    
    return {
        "context_docs": final_docs,
        "pii_mapping": pii_mappings,
        "current_step": f"Recuperados {len(final_docs)} fragmentos relevantes"
    }


def grade_documents(state: LegalGuardState):
    """Nodo condicional simulado en estado: Verifica si encontramos contexto."""
    docs = state.get("context_docs", [])
    if not docs:
        log_warn("nodo: grade_documents", "0 fragmentos. Activando reintento.")
        return {"retry_count": state.get("retry_count", 0) + 1}
    return {}


def generate_response(state: LegalGuardState):
    """Nodo: Genera respuesta usando hashes PII (Sin desencriptar aquí)."""
    log_sequence("nodo: generate_response", "Generando respuesta RAG")
    
    context = "\n\n".join([d["content"] for d in state.get("context_docs", [])])
    question = state["messages"][-1].content
    
    system_prompt = f"""Estarás respondiendo preguntas sobre contratos legales basados EXCLUSIVAMENTE en el siguiente contexto.
    No uses conocimientos externos. Algunas entidades (Nombres, DNI) aparecerán como hashes como <PERSON> o similar. 
    Usa exactamente los mismos hashes en tu respuesta.
    
    CONTEXTO:
    {context}
    """
    
    llm = _get_llm()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_step": "Respuesta generada (pre-deanonymize)"
    }


def deanonymize_output(state: LegalGuardState):
    """Nodo interceptor: Revierte los hashes de la respuesta final del LLM."""
    log_sequence("nodo: deanonymize_output", "Revertiendo hashes de Presidio")
    
    last_response = state["messages"][-1]
    pii_items_dict = state.get("pii_mapping", [])
    
    # Reconstrucción de la clase RecognizerResult requerida por presidio
    from presidio_analyzer import RecognizerResult
    pii_items = [
        RecognizerResult(
            entity_type=item["entity_type"], 
            start=item["start"], 
            end=item["end"], 
            score=1.0
        ) for item in pii_items_dict
    ]
    
    # Desencriptamos pasando el texto del LLM y los metadatos recuperados de Azure Search
    clear_text = deanonymize_text(last_response.content, pii_items)
    last_response.content = clear_text
    
    return {
        "current_step": "Respuesta segura retornada al usuario"
    }


def classify_intent(state: LegalGuardState):
    """Nodo Enrutador Principal: Clasifica si requiere RAG, Sandbox Matemático o Charla General."""
    log_sequence("nodo: classify_intent", "Evaluando intención del usuario")
    question = state["messages"][-1].content
    
    system_prompt = """Eres un enrutador estructurado. Responde SOLO con una de las siguientes tres palabras:
    - "CALCULO": Si la pregunta implica sumar, descontar fechas, o manipular tablas financieras explícitamente.
    - "GENERAL": Si es un saludo tipo hola, adiós, o una pregunta que no requiere contratos.
    - "LEGAL": Para todo lo demás (búsqueda de cláusulas, contratos, nombres, ubicaciones)."""
    
    llm = _get_llm()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    response = llm.invoke(messages)
    
    intent = response.content.strip().upper()
    log_info(f"Intención clasificada como: {intent}")
    
    return {"current_step": f"Clasificado: {intent}"}


def call_code_interpreter(state: LegalGuardState):
    """Nodo Mnemotécnico: Dispara el sandbox iterativo de Python sobre Azure Sessions."""
    log_sequence("nodo: call_code_interpreter", "Enviando al sandbox (Dynamic Sessions)")
    
    from src.tools.code_interpreter import ejecutar_analisis_financiero
    llm = _get_llm()
    
    # Invocamos al agente bindeado con la herramienta
    agent = llm.bind_tools([ejecutar_analisis_financiero])
    response = agent.invoke(state["messages"])
    
    return {
        "messages": [response],
        "current_step": "Ejecutando script de Pandas en la nube"
    }
