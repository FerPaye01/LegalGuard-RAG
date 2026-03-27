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
        
        # Modelo rápido "mini" para el Grader si estuviera disponible, 
        # sino usamos el mismo deployment (Opción B)
        self.fast_llm = self.llm 
        
        # Instanciar el motor de búsqueda híbrido que ya validamos
        self.search_engine = AzureSearchHybridEngine()
        
        # Gestor de Gobernanza e IA Responsable
        self.governance = GovernanceManager()
        
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

        # Configurar Flujo (Edges)
        workflow.set_entry_point("router")
        
        # Lógica Condicional del Router
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "legal": "retrieve",
                "general": "general_answer"
            }
        )
        
        workflow.add_edge("retrieve", "grade_docs")
        
        # Lógica Condicional del Grader
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

        return workflow.compile()

    # --- Lógica de los Nodos ---

    def router_node(self, state: AgentState):
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
        
        structured_llm = self.llm.with_structured_output(RouteQuery)
        
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
        log_sequence("Cerebro: Nodo Retriever", "Consultando Azure AI Search")
        question = state["messages"][-1].content
        filter_docs = state.get("filter_docs", [])
        
        # Consultar usando el motor híbrido (RRF) con filtro opcional
        results = self.search_engine.search_hybrid(
            query=question, 
            top_k=3, 
            filter_docs=filter_docs
        )
        
        return {"documents": results}

    def grader_node(self, state: AgentState):
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
        log_sequence("Cerebro: Nodo Generator", "Sintetizando respuesta con citas")
        question = state["messages"][-1].content
        docs = state["documents"]
        
        context = "\n\n".join([f"--- SOURCE: {d['source_file']} ---\n{d['content']}" for d in docs])
        
        system = """Eres LegalGuard, un asistente de IA experto en leyes. 
        Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.
        Si la información no está presente, admítelo.
        SIEMPRE cita el nombre del archivo de origen al final de tu respuesta."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Pregunta: {question} \n\n Contexto: {context}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format(question=question, context=context))
            answer = response.content
            
            # FILTRO 3: Gatekeeper Output (Content Safety + Anonimización PII Local)
            clean_answer, is_safe_output = self.governance.gatekeeper(answer, is_input=False)
            if not is_safe_output:
                 return {"answer": f"Lo siento, la respuesta generada fue bloqueada: {clean_answer}"}

            return {"answer": clean_answer}
        except Exception as e:
            if "content_filter" in str(e).lower():
                log_error("CRITICAL: El Filtro de Contenido bloqueó la generación final", e)
                return {"answer": "Lo siento, la respuesta a esta consulta fue bloqueada por las políticas de seguridad de contenido de Azure (Content Safety). Por favor, intenta reformular tu pregunta."}
            raise e

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
        if state["is_legal_query"]:
            return "legal"
        return "general"

    def grade_decision(self, state: AgentState):
        if state["is_relevant"]:
            return "useful"
        return "not_useful"

    def run(self, query: str, filter_docs: list = None):
        """Ejecuta el grafo completo para una pregunta."""
        inputs = {
            "messages": [HumanMessage(content=query)],
            "filter_docs": filter_docs or []
        }
        config = {"recursion_limit": 10}
        
        result = self.workflow.invoke(inputs, config=config)
        
        # Registro de Auditoría (Hito 3)
        self.governance.log_interaction(
            query=query,
            answer=result["answer"],
            documents=result.get("documents", []),
            metadata={"is_legal": result.get("is_legal_query")}
        )

        return {
            "answer": result["answer"],
            "documents": result.get("documents", []),
            "grader_counts": result.get("grader_counts", {})
        }

if __name__ == "__main__":
    # Test rápido de orquestación
    agent = LegalGuardAgent()
    
    print("\n--- TEST: CHARLA GENERAL ---")
    print(agent.run("Hola, ¿quién eres?"))
    
    print("\n--- TEST: PREGUNTA LEGAL (RAG) ---")
    print(agent.run("¿Cuáles son las penalidades por incumplimiento?"))
