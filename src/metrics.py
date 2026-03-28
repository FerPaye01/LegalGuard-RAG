import os
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from src.utils.logger import log_info, log_sequence, log_warn, log_error

load_dotenv()

# --- Configuración de Jueces IA para RAGAS ---
def get_ragas_judges():
    """Configura los modelos de Azure OpenAI como jueces de RAGAS."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    return llm, embeddings

def load_audit_samples(log_path="outputs/governance/audit_log.jsonl", max_samples=5):
    """Carga las últimas interacciones del log de auditoría para evaluar."""
    if not Path(log_path).exists():
        log_warn("No hay logs de auditoría para evaluar en " + log_path)
        return None

    samples = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            # Si piden más muestras de las que hay, tomamos todas
            count = min(max_samples, len(lines))
            
            # Tomamos las últimas interacciones
            for line in lines[-count:]:
                entry = json.loads(line)
                # RAGAS necesita: question, answer, contexts (lista de strings)
                samples.append({
                    "question": entry.get("query"),
                    "answer": entry.get("answer"),
                    "contexts": [doc.get("content", "") for doc in entry.get("context_used", []) 
                                 if doc.get("content")] or ["No context found"]
                })
    except Exception as e:
        log_error("Error cargando muestras para RAGAS", e)
        return None
    
    return samples

def run_evaluation(max_samples=5):
    """Ejecuta la evaluación RAGAS completa."""
    log_sequence("RAGAS", f"Iniciando evaluación de calidad con {max_samples} muestras (LLM-as-a-Judge)")
    
    samples = load_audit_samples(max_samples=max_samples)
    if not samples:
        return {"error": "No hay suficientes datos para evaluar."}

    # Convertir a Dataset de HuggingFace (formato RAGAS)
    dataset = Dataset.from_list(samples)
    
    llm, embeddings = get_ragas_judges()
    
    try:
        # Ejecutar evaluación
        # Nota: RAGAS usa el LLM para juzgar la fidelidad y relevancia
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings
        )
        
        df_results = result.to_pandas()
        log_info("RAGAS", f"Puntuaciones Promedio: {result}")
        
        # Guardar resultados históricos
        output_dir = Path("outputs/metrics")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar CSV
        df_results.to_csv(output_dir / "latest_ragas_report.csv", index=False)
        
        # Guardar JSON (Para Compliance)
        json_results = {
            "timestamp": datetime.now().isoformat(),
            "mean_scores": {k: float(v) for k, v in result.items()},
            "individual_samples": df_results.to_dict(orient="records")
        }
        with open(output_dir / "latest_ragas_report.json", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
            
        return json_results
    
    except Exception as e:
        log_error("Error durante evaluación RAGAS", e)
        return {"error": str(e)}


# Evaluación Lite: Una sola respuesta (para modo Real-Time)
def eval_single_response(question: str, answer: str, contexts: list) -> dict:
    """Evalúa la fidelidad y relevancia de UNA sola respuesta del RAG."""
    log_sequence("RAGAS Lite", "Evaluando fidelidad de respuesta individual")
    
    if not question or not answer:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "status": "incomplete"}
    
    samples = [{
        "question": question,
        "answer": answer,
        "contexts": contexts or ["No context found"]
    }]
    
    dataset = Dataset.from_list(samples)
    llm, embeddings = get_ragas_judges()
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings
        )
        
        scores = {k: float(v) for k, v in result.items()}
        scores["status"] = "ok"
        log_info(f"RAGAS Lite: Faithfulness={scores.get('faithfulness', 0):.2%}")
        return scores
    except Exception as e:
        log_warn(f"RAGAS Lite falló: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "status": f"error: {str(e)[:50]}"}

if __name__ == "__main__":
    results = run_evaluation()
    print("\n--- RESULTADOS RAGAS (0.0 a 1.0) ---")
    print(results)
