"""
Grafo Maestro (Orquestador Principal) del LegalGuard RAG.
Conecta los nodos de LangGraph bajo la estructura de Opción B.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.orchestration.state import LegalGuardState
from src.orchestration.nodes import (
    retrieve_contracts, 
    generate_response, 
    deanonymize_output, 
    rewrite_query,
    classify_intent,
    call_code_interpreter
)
from src.utils.logger import log_sequence, log_warn


def grade_route(state: LegalGuardState):
    """Enrutador: Decide si continuar a generar, buscar de nuevo o finalizar (Circuit Breaker)."""
    context = state.get("context_docs", [])
    retry_count = state.get("retry_count", 0)
    
    if not context:
        if retry_count >= 1:
            log_warn("enrutador: grade_route", "Circuit Breaker activado (Reintentos > 1)")
            return "end" # Directo a END para que responda que no encontró nada
        else:
            log_warn("enrutador: grade_route", f"Sin contexto. Reintentos = {retry_count}. Mandando a rewrite.")
            return "rewrite"
    return "generate"


def router_classify(state: LegalGuardState):
    """Enrutador Principal: Dirige el flujo basado en la intención."""
    # En nuestro diseño, classify_intent deja el resultado en el estado como current_step
    step = state.get("current_step", "")
    
    if "CALCULO" in step:
        return "interpreter"
    elif "LEGAL" in step:
        return "retrieve"
    else:  # GENERAL
        return "generate"


def build_graph():
    """Compila el orquestador maestro con persistencia en memoria y chequeos de recursos."""
    log_sequence("langgraph: build_graph", "Ensamblando StateGraph de LegalGuard RAG")
    
    builder = StateGraph(LegalGuardState)

    # Añadir nodos
    builder.add_node("classify", classify_intent)
    builder.add_node("retrieve", retrieve_contracts)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("interpreter", call_code_interpreter)
    builder.add_node("generate", generate_response)
    builder.add_node("deanonymize", deanonymize_output)

    # Definir flujo
    builder.add_edge(START, "classify")
    
    # Enrutamiento principal desde Classify
    builder.add_conditional_edges(
        "classify",
        router_classify,
        {
            "retrieve": "retrieve",
            "interpreter": "interpreter",
            "generate": "generate"
        }
    )
    
    # Enrutamiento (Reflexivo RAG)
    builder.add_conditional_edges(
        "retrieve",
        grade_route,
        {
            "rewrite": "rewrite",
            "generate": "generate",
            "end": END
        }
    )
    
    # Bucle seguro de re-escritura (Solo 1 intento permitido)
    builder.add_edge("rewrite", "retrieve")
    
    # El interprete de código calcula y se lo pasa a generar
    builder.add_edge("interpreter", "generate")
    
    # Después de generar, obligatoriamente se intercepta el PII
    builder.add_edge("generate", "deanonymize")
    builder.add_edge("deanonymize", END)

    # Persistencia efímera para interrupción de hilos y memoria
    checkpointer = MemorySaver()
    
    # Compilación final. Al ejecutar app.invoke(), DEBES pasar config={"recursion_limit": 5}
    app = builder.compile(checkpointer=checkpointer)
    
    return app
