import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

from src.orchestration.graph import build_graph
from src.utils.logger import log_info, log_error

app = FastAPI(title="LegalGuard RAG - API de Orquestación")

# Habilitar CORS para el frontend (Streamlit u otros)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia única del grafo (Maestro)
graph_app = build_graph()

@app.get("/health")
async def health():
    return {"status": "ok", "service": "LegalGuard RAG"}

@app.post("/ask")
async def ask_legalguard(request: Request):
    """
    Endpoint que emite Eventos de Servidor (SSE) mientras el grafo procesa.
    """
    body = await request.json()
    question = body.get("question", "")
    thread_id = body.get("thread_id", "default_user")

    if not question:
        return {"error": "No se proporcionó ninguna pregunta."}

    async def event_generator():
        log_info("api:sse", f"Iniciando flujo SSE para: {question[:50]}...")
        
        inputs = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}

        try:
            # Iteramos sobre el flujo asíncrono del grafo
            async for output in graph_app.astream(inputs, config=config):
                # Cada 'output' es un diccionario {nombre_nodo: estado_parcial}
                for node_name, state_update in output.items():
                    
                    # Estructuramos el evento para el frontend
                    event_data = {
                        "node": node_name,
                        "status": state_update.get("current_step", "Procesando..."),
                        "partial_answer": ""
                    }
                    
                    # Si el nodo produjo mensajes, enviamos el último
                    if "messages" in state_update and state_update["messages"]:
                        event_data["partial_answer"] = state_update["messages"][-1].content
                    
                    yield {
                        "event": "thought",
                        "data": json.dumps(event_data)
                    }
            
            yield {"event": "finish", "data": "Streaming completado"}

        except Exception as e:
            log_error("api:sse", e)
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(e)})
            }

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
