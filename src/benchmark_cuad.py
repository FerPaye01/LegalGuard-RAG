import os
import json
import pandas as pd
from dotenv import load_dotenv
from src.agent import LegalGuardAgent
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

load_dotenv()

# --- CONFIGURACIÓN DEL BENCHMARK ---
# Para la demo, seleccionamos un subconjunto representativo del dataset ContractNLI
BENCHMARK_FILE = "data/contractnli/contract-nli/dev.json"
NUM_DOCS_TO_TEST = 2 # Limitar para no quemar tokens innecesariamente

# Preguntas estándar basadas en los labels de ContractNLI (NDA)
CUAD_QUESTIONS = {
    "nda-11": "¿Cuál es la ley aplicable (Governing Law) que rige este contrato?",
    "nda-10": "¿Cuál es el periodo de terminación o expiración de este acuerdo?",
    "nda-16": "¿Se permite la cesión (assignment) de este contrato a terceros sin consentimiento?",
    "nda-1": "¿Existe alguna cláusula de no competencia (non-compete) incluida?",
    "nda-7": "¿Cómo se define la Información Confidencial en el texto?"
}

def load_benchmark_data():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"❌ Error: No se encuentra el archivo {BENCHMARK_FILE}")
        return []
    
    try:
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["documents"][:NUM_DOCS_TO_TEST]
    except Exception as e:
        print(f"❌ Error leyendo JSON: {e}")
        return []

def run_benchmarking():
    print("🚀 Iniciando Auditoría Técnica de Precisión (Benchmark CUAD)...")
    agent = LegalGuardAgent()
    docs = load_benchmark_data()
    
    if not docs:
        print("⚠️ No hay documentos para evaluar.")
        return

    eval_data = []

    for doc in docs:
        filename = doc["file_name"]
        print(f"\n📂 Evaluando documento: {filename}")
        
        # Obtenemos las verdades humanas de las anotaciones de ContractNLI
        annotations = doc["annotation_sets"][0]["annotations"]
        
        for label, question in CUAD_QUESTIONS.items():
            if label in annotations:
                human_label = annotations[label]["choice"] # "Entailment", "Contradiction", "NotMentioned"
                
                print(f"❓ Preguntando: {question}")
                
                # Ejecutamos el agente (Filtrando por este documento específico)
                result = agent.run(query=question, filter_docs=[filename])
                
                # Preparamos el registro para RAGAS
                eval_data.append({
                    "question": question,
                    "answer": result.get("answer", "Sin respuesta"),
                    "contexts": [d["content"] for d in result.get("documents", [])],
                    "ground_truth": f"El contrato indica un estado de {human_label} para la cláusula {label}."
                })

    if not eval_data:
        print("⚠️ No se extrajeron datos para evaluar.")
        return

    # --- EVALUACIÓN RAGAS ---
    print("\n⚖️ Enviando resultados al Juez IA (GPT-4o) para validación matemática...")
    
    # Preparamos el dataset para RAGAS
    # Nota: pregunta/respuesta/contexto son obligatorios
    data_dict = {
        "question": [d["question"] for d in eval_data],
        "answer": [d["answer"] for d in eval_data],
        "contexts": [d["contexts"] for d in eval_data],
        "ground_truth": [d["ground_truth"] for d in eval_data]
    }
    dataset = Dataset.from_dict(data_dict)
    
    try:
        # El comando evaluate calcula las métricas de fidelidad y relevancia
        result_scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision]
        )
        
        # --- REPORTE FINAL ---
        print("\n" + "="*40)
        print("   📊 RESULTADOS BENCHMARK CUAD / CONTRACTNLI   ")
        print("="*40)
        print(result_scores)
        print("="*40)
        
        # Guardar en CSV para la Demo
        os.makedirs("outputs", exist_ok=True)
        df = result_scores.to_pandas()
        df.to_csv("outputs/benchmark_results.csv", index=False)
        print(f"✅ Reporte detallado guardado en: outputs/benchmark_results.csv")
    except Exception as e:
        print(f"❌ Error en la evaluación RAGAS: {e}")

if __name__ == "__main__":
    run_benchmarking()
