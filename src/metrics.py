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
            # Tomamos las últimas interacciones
            for line in lines[-max_samples:]:
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

def run_evaluation():
    """Ejecuta la evaluación RAGAS completa."""
    log_sequence("RAGAS", "Iniciando evaluación de calidad (LLM-as-a-Judge)")
    
    samples = load_audit_samples()
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
        log_error("Fallo crítico en evaluación RAGAS", e)
        return {"error": str(e)}

if __name__ == "__main__":
    results = run_evaluation()
    print("\n--- RESULTADOS RAGAS (0.0 a 1.0) ---")
    print(results)
