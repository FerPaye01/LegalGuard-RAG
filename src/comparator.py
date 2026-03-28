import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from src.retrieval.search_engine import AzureSearchHybridEngine
from src.utils.logger import log_info, log_sequence, log_warn, log_error

load_dotenv()

def compare_contract_versions(doc_v1_name: str, doc_v2_name: str) -> dict:
    """
    Compara dos versiones de un contrato recuperando fragmentos de Azure AI Search
    y detectando cambios mediante GPT-4o-mini.
    """
    log_sequence("Comparador de Versiones", f"Comparando {doc_v1_name} vs {doc_v2_name}")
    
    search_engine = AzureSearchHybridEngine()
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )
    
    # Recuperar fragmentos de ambas versiones
    docs_v1 = search_engine.search_hybrid("cláusulas y condiciones generales", top_k=8, filter_docs=[doc_v1_name])
    docs_v2 = search_engine.search_hybrid("cláusulas y condiciones generales", top_k=8, filter_docs=[doc_v2_name])
    
    if not docs_v1 and not docs_v2:
        return {"error": "No se encontraron fragmentos para ninguno de los documentos en el índice."}
    
    if not docs_v1:
        return {"error": f"No se encontraron fragmentos del documento: {doc_v1_name}"}
    
    if not docs_v2:
        return {"error": f"No se encontraron fragmentos del documento: {doc_v2_name}"}
    
    text_v1 = "\n---\n".join([d["content"] for d in docs_v1])
    text_v2 = "\n---\n".join([d["content"] for d in docs_v2])
    
    prompt = f"""Eres un auditor legal experto en comparación de contratos.

Compara estas dos versiones del mismo contrato y devuelve SOLO un JSON válido con la siguiente estructura:
{{
  "resumen": "Una frase resumiendo la gravedad de los cambios",
  "cambios": [
    {{
      "tipo": "modificado|nuevo|eliminado",
      "clausula": "nombre o número de la cláusula",
      "antes": "texto en la versión anterior o null si es nuevo",
      "despues": "texto en la nueva versión o null si fue eliminado",
      "impacto": "alto|medio|bajo"
    }}
  ]
}}

--- VERSIÓN ANTERIOR ({doc_v1_name}) ---
{text_v1[:3000]}

--- VERSIÓN NUEVA ({doc_v2_name}) ---
{text_v2[:3000]}

Responde ÚNICAMENTE con el JSON, sin texto adicional, sin markdown."""
    
    try:
        response = llm.invoke(prompt)
        import json
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        log_info(f"Comparación completada: {len(result.get('cambios', []))} cambios detectados")
        return result
    except json.JSONDecodeError as e:
        log_warn(f"Respuesta del LLM no fue JSON válido: {e}")
        return {
            "resumen": "Comparación completada (respuesta no estructurada)",
            "cambios": [{"tipo": "modificado", "clausula": "General", "antes": text_v1[:300], "despues": text_v2[:300], "impacto": "medio"}]
        }
    except Exception as e:
        log_error("Error en el comparador de versiones", e)
        return {"error": str(e)}
