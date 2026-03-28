import os
import json
from typing import Annotated, List, Union, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.retrieval.search_engine import AzureSearchHybridEngine
from src.utils.logger import log_info, log_sequence, log_warn, log_error
from src.governance import GovernanceManager
from src.telemetry import NodeTimer, track_node_latency, init_application_insights, track_usage
import tiktoken

load_dotenv()

# --- Definición del Estado del Agente ---
class AgentState(TypedDict):
    """
    Estado compartido que viaja entre los nodos del Grafo.
    """
    messages: Annotated[List[BaseMessage], "Historia de la conversación"]
    documents: List[dict] # Fragmentos recuperados de Azure AI Search
    filter_docs: List[str] # Lista de archivos seleccionados por el usuario
    is_legal_query: bool  # Flag del Router
    is_relevant: bool     # Flag del Grader
    answer: str          # Respuesta final generada
    grader_counts: dict  # Metadatos para la UI (total_found, total_relevant)
    persona: str         # Perfil profesional del usuario (Legal, Financiero, Salud)
    code_output: str     # Resultado de la Calculadora Legal (Dynamic Sessions)

# --- Modelos de Datos para Estructuración ---
class RouteQuery(BaseModel):
    """Esquema para que el LLM decida si la pregunta es legal o no."""
    datasource: str = Field(
        ...,
        description="El destino de la pregunta. Puede ser 'legal_search' o 'general_chat'."
    )

class GradeDocument(BaseModel):
    """Esquema para calificar si un documento es relevante para la pregunta."""
    binary_score: str = Field(
        ...,
        description="¿Es el documento relevante para la pregunta? 'yes' o 'no'"
    )

# --- Implementación de Nodos ---

class LegalGuardAgent:
    def __init__(self):
        # Clientes de Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0
        )
        
        # GPT-4o-mini para tareas rápidas (Grader, Router, Calculadora)
        mini_deployment = os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"))
        self.fast_llm = AzureChatOpenAI(
            azure_deployment=mini_deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0
        )
        
        # Instanciar el motor de búsqueda híbrido que ya validamos
        self.search_engine = AzureSearchHybridEngine()
        
        # Gestor de Gobernanza e IA Responsable
        self.governance = GovernanceManager()
        
        # Cronómetro de Nodos (Telemetría)
        self.timer = NodeTimer()
        
        # Inicializar Application Insights si está configurado
        init_application_insights()
        
        # Construcción del Grafo
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Definir Nodos
        workflow.add_node("router", self.router_node)
        workflow.add_node("retrieve", self.retriever_node)
        workflow.add_node("grade_docs", self.grader_node)
        workflow.add_node("generate", self.generator_node)
        workflow.add_node("general_answer", self.general_node)
        workflow.add_node("calculate", self.calculator_node)

        # Configurar Flujo (Edges)
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "legal": "retrieve",
                "general": "general_answer",
                "math": "calculate"
            }
        )
        
        workflow.add_edge("retrieve", "grade_docs")
        
        workflow.add_conditional_edges(
            "grade_docs",
            self.grade_decision,
            {
                "useful": "generate",
                "not_useful": "general_answer"
            }
        )
        
        workflow.add_edge("generate", END)
        workflow.add_edge("general_answer", END)
        workflow.add_edge("calculate", END)

        return workflow.compile()

    # --- Lógica de los Nodos ---

    def router_node(self, state: AgentState):
        self.timer.start("router")
        log_sequence("Gobernanza", "Validando seguridad de la consulta...")
        question = state["messages"][-1].content
        
        # FILTRO 1: Gatekeeper (Content Safety Input)
        filtered_text, is_safe = self.governance.gatekeeper(question, is_input=True)
        if not is_safe:
            return {
                "answer": f"Lo siento, tu consulta ha sido bloqueada: {filtered_text}",
                "is_legal_query": False
            }

        log_sequence("Cerebro: Nodo Router", "Clasificando intención")
        
        structured_llm = self.fast_llm.with_structured_output(RouteQuery)
        
        # Detectar intención matemática (para Calculadora Legal)
        math_keywords = ["cuánto", "cuanto", "suma", "total", "calcula", "monto", "penalidad total", "sumar", "promedio", "cuantos", "cuántos"]
        if any(kw in question.lower() for kw in math_keywords):
            log_info("Detectada intención matemática → enrutando a Calculadora Legal")
            return {"is_legal_query": True}
        
        system = """Eres un experto en enrutamiento legal y documental. 
        Si el usuario pregunta algo sobre leyes, contratos, NDAs, plazos, o sobre el CONTENIDO de cualquier documento o procedimiento (SOP) cargado, 
        responde con 'legal_search'. 
        Si es un saludo, charla general o una pregunta que no requiere consultar los documentos, responde con 'general_chat'."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}")
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"question": question})
        
        return {
            "is_legal_query": result.datasource == "legal_search"
        }

    def retriever_node(self, state: AgentState):
        self.timer.stop("router")
        self.timer.start("retriever")
        log_sequence("Cerebro: Nodo Retriever", "Consultando Azure AI Search")
        question = state["messages"][-1].content
        filter_docs = state.get("filter_docs", [])
        
        # Consultar usando el motor híbrido (RRF) con filtro opcional
        results = self.search_engine.search_hybrid(
            query=question, 
            top_k=5,
            filter_docs=filter_docs
        )
        
        return {"documents": results}

    def grader_node(self, state: AgentState):
        self.timer.stop("retriever")
        self.timer.start("grader")
        log_sequence("Cerebro: Nodo Grader", "Validando relevancia de TODOS los fragmentos (Opción A)")
        question = state["messages"][-1].content
        docs = state["documents"]
        
        if not docs:
            log_warn("No se recuperaron documentos para calificar.")
            return {"is_relevant": False, "grader_counts": {"total_found": 0, "total_relevant": 0}}

        # FILTRO 2: Umbral de Confianza Estricto (0.7)
        # Para RRF, el score máximo es ~0.033. Mapeamos 0.7 -> 0.023 aprox.
        # Si el match es débil, el sistema debe ser honesto.
        top_score = docs[0].get("score", 0) 
        UMBRAL_ESTRICTO = 0.015 # Calibrado para mayor cobertura en documentos individuales (RRF 0.015+)
        
        if top_score < UMBRAL_ESTRICTO:
             log_warn(f"Confianza insuficiente (RRF {top_score:.4f} < {UMBRAL_ESTRICTO}). Bloqueando RAG.")
             return {"is_relevant": False, "grader_counts": {"total_found": len(docs), "total_relevant": 0}}

        structured_llm = self.fast_llm.with_structured_output(GradeDocument)
        
        system = """Eres un experto calificador de relevancia documental y legal. 
        Evalúa si el siguiente fragmento del documento o procedimiento (SOP) contiene información ÚTIL y DIRECTA para responder la pregunta del usuario. 
        Solo responde 'yes' si el fragmento aporta evidencia real basada estrictamente en el texto, en caso de duda o si es ruido irrelevante para la consulta, responde 'no'."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Pregunta: {question} \n\n Fragmento a Evaluar: {context}")
        ])
        
        chain = prompt | structured_llm
        
        # Evaluar cada documento en paralelo
        relevant_docs = []
        try:
            # Preparamos las entradas para el batch
            inputs = [{"question": question, "context": doc["content"]} for doc in docs]
            # Usamos batch para paralelizar las llamadas al LLM
            scores = chain.batch(inputs)
            
            for doc, score in zip(docs, scores):
                if score.binary_score == "yes":
                    relevant_docs.append(doc)
                    log_info(f"✅ Fragmento APROBADO: {doc['source_file']}")
                else:
                    log_info(f"❌ Fragmento RECHAZADO (Irrelevante): {doc['source_file']}")
            
            is_relevant = len(relevant_docs) > 0
            
            return {
                "documents": relevant_docs, # Sobrescribimos con los puros
                "is_relevant": is_relevant,
                "grader_counts": {
                    "total_found": len(docs),
                    "total_relevant": len(relevant_docs)
                }
            }
        except Exception as e:
            if "content_filter" in str(e).lower():
                log_error("CRITICAL: El Filtro de Contenido de Azure bloqueó el fragmento", e)
                return {"is_relevant": False, "grader_counts": {"total_found": len(docs), "total_relevant": 0}}
            raise e

    def generator_node(self, state: AgentState):
        self.timer.stop("grader")
        self.timer.start("generator")
        log_sequence("Cerebro: Nodo Generator", "Sintetizando respuesta con citas HTML y anti-contradicciones")
        question = state["messages"][-1].content
        docs = state["documents"]
        persona = state.get("persona", "Orchestrator")
        
        # Instrucciones de rol por persona
        persona_instructions = {
            "Legal": "Enfoca tu respuesta en obligaciones, derechos, cláusulas contractuales y posible jurisprudencia aplicable. Usa terminología jurídica precisa.",
            "Financiero": "Destaca montos, penalidades económicas, plazos de pago, condiciones financieras y proyecciones de costo. Habla en cifras cuando sea posible.",
            "Salud": "Prioriza normativas sanitarias, protocolos de bioseguridad, regulaciones de salud pública y procedimientos clínicos relevantes.",
            "Orchestrator": "Proporciona una visión equilibrada y completa del documento, balanceando aspectos legales, financieros y operativos."
        }
        role_instruction = persona_instructions.get(persona, persona_instructions["Orchestrator"])
        
        # Contexto enriquecido con fecha de ingesta para resolución de contradicciones
        context = "\n\n".join([
            f"[Documento: {d['source_file']}]\n[Fecha de Ingesta: {d.get('upload_date', 'Desconocida')}]\nTexto: {d['content']}"
            for d in docs
        ])
        
        system = f"""Eres LegalGuard, un asistente legal de IA de nivel empresarial. Sigue estas reglas SIN EXCEPCIÓN:

**ROL DEL USUARIO: {persona.upper()}** — {role_instruction}

1.  **BASE DOCUMENTAL**: Responde basándote ÚNICAMENTE en el contexto proporcionado. Si la info no está, admítelo: 'No encontré información suficiente en los documentos para responder esto.'

2.  **CITAS OBLIGATORIAS (MUY IMPORTANTE)**: Cuando hagas una afirmación clave basada en el documento, DEBES envolver esa frase en esta etiqueta HTML exacta:
    `<span class="cite-highlight" data-fragment="[Cita textual exacta del fragmento original]">tu frase generada</span>`
    Ejemplo: El contrato establece que <span class="cite-highlight" data-fragment="La multa por incumplimiento será de $5,000 dentro de los 30 días.">el incumplimiento acarrea una multa de $5,000</span>.
    NO uses corchetes [1] ni notas al pie. USA SOLO esta etiqueta HTML.

3.  **ANTI-CONTRADICCIONES**: Si detectas información contradictoria entre dos fragmentos:
    a) Señálalo explícitamente: '⚠️ Se detectó una contradicción entre documentos.'
    b) Responde ÚNICAMENTE basándote en el documento con la [Fecha de Ingesta] MÁS RECIENTE.
    c) Explica brevemente por qué descartaste el otro: 'El documento X (fecha Y) fue descartado por ser anterior.'"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Pregunta: {question} \n\n Contexto con fechas:\n{context}")
        ])
        
        try:
            full_prompt = prompt.format(question=question, context=context)
            response = self.llm.invoke(full_prompt)
            answer = response.content
            
            # --- TELEMETRÍA DE TOKENS (Hito 15) ---
            try:
                encoding = tiktoken.get_encoding("o200k_base")
                in_tokens = len(encoding.encode(full_prompt))
                out_tokens = len(encoding.encode(answer))
                track_usage(in_tokens, out_tokens, os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"))
            except: pass
            
            # FILTRO 3: Gatekeeper Output (Content Safety + Anonimización PII Local)
            clean_answer, is_safe_output = self.governance.gatekeeper(answer, is_input=False)
            if not is_safe_output:
                 self.timer.stop("generator")
                 return {"answer": f"Lo siento, la respuesta generada fue bloqueada: {clean_answer}"}

            self.timer.stop("generator")
            return {"answer": clean_answer}
        except Exception as e:
            if "content_filter" in str(e).lower():
                log_error("CRITICAL: El Filtro de Contenido bloqueó la generación final", e)
                return {"answer": "Lo siento, la respuesta a esta consulta fue bloqueada por las políticas de seguridad de contenido de Azure (Content Safety). Por favor, intenta reformular tu pregunta."}
            raise e

    def calculator_node(self, state: AgentState):
        """Nodo Calculadora Legal: ejecuta Python en Azure Dynamic Sessions."""
        self.timer.stop("router")
        self.timer.start("calculator")
        log_sequence("Cerebro: Nodo Calculadora", "Ejecutando código Python en Dynamic Sessions (Azure)")
        question = state["messages"][-1].content
        
        # Preámbulo inmutable según el Skill de Dynamic Sessions
        PREAMBULO = """
import pandas as pd
import re
import json

# Función de limpieza de monedas (evita NaN silencioso de errors='coerce')
limpiar_moneda = lambda val: float(re.sub(r'[^\\d.]', '', str(val))) if re.sub(r'[^\\d.]', '', str(val)) else 0.0
"""
        
        pool_endpoint = os.getenv("AZURE_CONTAINER_APP_SESSION_POOL")
        
        if not pool_endpoint:
            self.timer.stop("calculator")
            return {"answer": "⚠️ La Calculadora Legal requiere Azure Container Apps. Configura `AZURE_CONTAINER_APP_SESSION_POOL`.", "code_output": ""}
        
        try:
            from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
            
            repl_tool = SessionsPythonREPLTool(pool_management_endpoint=pool_endpoint)
            
            # Generar el script Python con GPT-4o-mini
            code_prompt = f"""El usuario pregunta: '{question}'

Genera un script Python usando SOLO la función limpiar_moneda() para limpiar valores monetarios.
Usa SIEMPRE pd.to_numeric(..., errors='raise') NUNCA errors='coerce'.
El script debe terminar con un print() del resultado en formato JSON: {{'resultado': valor, 'descripcion': 'texto'}}
Genera SOLO el código Python, sin explicaciones ni markdown."""
            
            code_response = self.fast_llm.invoke(code_prompt)
            generated_code = code_response.content.strip().replace("```python", "").replace("```", "")
            
            # Empaquetar con el preámbulo inmutable
            final_code = PREAMBULO + "\n" + generated_code
            
            # Ejecutar en la Micro-VM de Azure
            log_info(f"Ejecutando en Dynamic Sessions:\n{final_code[:200]}...")
            result_str = repl_tool.invoke(final_code)
            
            # --- TELEMETRÍA DE TOKENS (Hito 15) ---
            try:
                encoding = tiktoken.get_encoding("o200k_base")
                in_tokens = len(encoding.encode(code_prompt))
                out_tokens = len(encoding.encode(generated_code))
                track_usage(in_tokens, out_tokens, "gpt-4o-mini-calc")
            except: pass
            
            self.timer.stop("calculator")
            answer = f"""🧮 **Resultado de la Calculadora Legal**

{result_str}

<details><summary>Ver código Python ejecutado</summary>

```python
{final_code}
```

</details>"""
            return {"answer": answer, "code_output": str(result_str)}
            
        except Exception as e:
            log_error("Error en Calculadora Legal", e)
            self.timer.stop("calculator")
            return {"answer": f"Error en la Calculadora Legal: {str(e)[:200]}", "code_output": "error"}

    def general_node(self, state: AgentState):
        log_sequence("Cerebro: Nodo General", "Charla no-legal")
        question = state["messages"][-1].content
        
        if not state.get("is_legal_query", False):
            msg = "¡Hola! Soy LegalGuard. Puedo ayudarte a analizar tus contratos. ¿Tienes alguna duda específica sobre un documento legal?"
        else:
            msg = "Mi análisis se limita estrictamente al contenido de los contratos proporcionados y no he encontrado información suficientemente fiable para responder a eso. ¿Te gustaría subir un nuevo documento o intentar reformular?"
            
        return {"answer": msg}

    # --- Funciones de Lógica de Aristas (Edges) ---

    def route_decision(self, state: AgentState):
        question = state["messages"][-1].content
        math_keywords = ["cuánto", "cuanto", "suma", "total", "calcula", "monto", "penalidad total", "sumar", "promedio", "cuántos", "cuantos"]
        if any(kw in question.lower() for kw in math_keywords):
            return "math"
        if state["is_legal_query"]:
            return "legal"
        return "general"

    def grade_decision(self, state: AgentState):
        if state["is_relevant"]:
            return "useful"
        return "not_useful"

    def run(self, query: str, filter_docs: list = None, persona: str = "Orchestrator"):
        """Ejecuta el grafo completo para una pregunta."""
        self.timer.reset()
        
        inputs = {
            "messages": [HumanMessage(content=query)],
            "filter_docs": filter_docs or [],
            "persona": persona,
            "code_output": ""
        }
        config = {"recursion_limit": 10}
        
        result = self.workflow.invoke(inputs, config=config)
        
        # Capturar telemetría del grafo
        telemetry_report = self.timer.get_report()
        track_node_latency(telemetry_report)
        
        # Registro de Auditoría
        self.governance.log_interaction(
            query=query,
            answer=result["answer"],
            documents=result.get("documents", []),
            metadata={
                "is_legal": result.get("is_legal_query"),
                "persona": persona,
                "telemetry": telemetry_report
            }
        )

        return {
            "answer": result["answer"],
            "documents": result.get("documents", []),
            "grader_counts": result.get("grader_counts", {}),
            "telemetry": telemetry_report,
            "code_output": result.get("code_output", "")
        }

if __name__ == "__main__":
    # Test rápido de orquestación
    agent = LegalGuardAgent()
    
    print("\n--- TEST: CHARLA GENERAL ---")
    print(agent.run("Hola, ¿quién eres?"))
    
    print("\n--- TEST: PREGUNTA LEGAL (RAG) ---")
    print(agent.run("¿Cuáles son las penalidades por incumplimiento?"))
