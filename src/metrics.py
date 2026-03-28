import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from src.utils.logger import log_debug, log_info, log_sequence, log_warn, log_error

load_dotenv()

# --- Directorios de Salida ---
DIR_OUTPUTS = Path("outputs/metrics")
DIR_LOGS = Path("outputs/logs")
DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
DIR_LOGS.mkdir(parents=True, exist_ok=True)

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

def configure_bilingual_metrics(metrics: list):
    """
    Configura las métricas de Ragas para soportar evaluación bilingüe (Español-Inglés).
    Inyecta instrucciones en los prompts internos para que el Juez IA entienda
    la equivalencia semántica entre ambos idiomas.
    """
    from ragas.metrics import Faithfulness, AnswerRelevancy
    
    instruction_suffix = (
        "\n\nIMPORTANT: The 'answer' may be in SPANISH while the 'context' is in ENGLISH. "
        "Prioritize semantic equivalence over literal matching. If a claim in Spanish "
        "is supported by the English context, it should be marked as FAITHFUL. "
        "Similarly, for relevancy, ensure the answer addresses the question regardless of language mix."
    )

    for m in metrics:
        try:
            prompts = m.get_prompts()
            for p_key, p_obj in prompts.items():
                if hasattr(p_obj, "instruction"):
                    p_obj.instruction += instruction_suffix
                elif hasattr(p_obj, "description"):
                    p_obj.description += instruction_suffix
            m.set_prompts(**prompts)
            log_info("RAGAS", f"Prompt bilingüe inyectado en métrica: {m.name}")
        except Exception as e:
            log_warn(f"No se pudo inyectar prompt bilingüe en {m.name}: {e}")

def strip_html(text: str) -> str:
    """Elimina etiquetas HTML y limpia el texto para el evaluador."""
    import re
    if not text: return ""
    # Quitar etiquetas span de citas y otras
    clean = re.sub(r'<[^>]+>', '', text)
    # Limpiar espacios extra
    return " ".join(clean.split())

# ════════════════════════════════════════════════════════════════════════════
# REGISTRO DE CONSULTAS (Trazabilidad y Auditoría)
# ════════════════════════════════════════════════════════════════════════════

def registrar_consulta(
    pregunta: str,
    respuesta: str,
    fragmentos: list,
    fuente: str,
    score_confianza: float,
    dominio: str = "legal"
) -> dict:
    """
    Registra cada consulta al RAG para trazabilidad y auditoría visual en el Dashboard.
    Guarda en outputs/logs/consultas.jsonl
    """
    registro = {
        "timestamp"       : datetime.now().isoformat(),
        "pregunta"        : pregunta,
        "respuesta"       : respuesta,
        "fragmentos"      : fragmentos[:3],   # primeros 3 fragmentos
        "fuente"          : fuente,
        "score_confianza" : round(score_confianza, 3),
        "dominio"         : dominio,
        "tiene_fuente"    : bool(fuente),
        "es_confiable"    : score_confianza >= 0.5,
    }

    ruta_log = DIR_LOGS / "consultas.jsonl"
    with open(ruta_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return registro

def cargar_historial() -> list:
    """Carga el historial completo de consultas registradas."""
    ruta_log = DIR_LOGS / "consultas.jsonl"
    if not ruta_log.exists():
        return []

    registros = []
    with open(ruta_log, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    pass
    return registros

def calcular_stats_historial(registros: list) -> dict:
    """Calcula estadísticas agregadas del historial de consultas."""
    if not registros:
        return {
            "total_consultas"     : 0,
            "con_fuente_pct"      : 0,
            "confiables_pct"      : 0,
            "score_promedio"      : 0,
            "por_dominio"         : {},
            "ultima_consulta"     : None,
        }

    total        = len(registros)
    con_fuente   = sum(1 for r in registros if r.get("tiene_fuente"))
    confiables   = sum(1 for r in registros if r.get("es_confiable"))
    scores       = [r.get("score_confianza", 0) for r in registros]
    por_dominio  = {}

    for r in registros:
        d = r.get("dominio", "legal")
        por_dominio[d] = por_dominio.get(d, 0) + 1

    return {
        "total_consultas"  : total,
        "con_fuente_pct"   : round(con_fuente / total * 100),
        "confiables_pct"   : round(confiables / total * 100),
        "score_promedio"   : round(sum(scores) / len(scores), 3) if scores else 0,
        "por_dominio"      : por_dominio,
        "ultima_consulta"  : registros[-1]["timestamp"] if registros else None,
    }

# ════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN RAGAS (Benchmark vs Auditoría)
# ════════════════════════════════════════════════════════════════════════════

def preparar_dataset_cuad(n_muestras: int = 20) -> Optional[list]:
    """Prepara los pares pregunta-respuesta del dataset CUAD para benchmarking."""
    ruta_cuad = Path("data/cuad/cuad_muestra_50.json")
    if not ruta_cuad.exists():
        log_warn("No se encontró el dataset CUAD en " + str(ruta_cuad))
        return None

    try:
        with open(ruta_cuad, "r", encoding="utf-8") as f:
            contratos = json.load(f)

        pares = []
        for contrato in contratos[:n_muestras]:
            contexto  = contrato.get("context", "")
            presentes = contrato.get("clausulas_presentes", [])

            for clausula in presentes[:2]:
                pares.append({
                    "question"          : clausula["clausula"],
                    "ground_truth"      : clausula["texto"],
                    "contexts"          : [contexto[:1500]], # Token limit safety
                    "titulo"            : contrato.get("titulo", ""),
                })
                if len(pares) >= n_muestras: break
            if len(pares) >= n_muestras: break
        return pares
    except Exception as e:
        log_error("Error preparando dataset CUAD", e)
        return None

def run_evaluation(samples: list = None, max_samples: int = 5) -> dict:
    """
    Ejecuta la evaluación RAGAS. 
    Si no se pasan muestras, intenta cargar los logs de auditoria en vivo.
    Detecta automáticamente si hay 'ground_truth' para habilitar Precision/Recall.
    """
    if not samples:
        # Cargar de logs de auditoria (Modo Auditoría en Vivo)
        log_sequence("RAGAS Audit", f"Evaluando últimas {max_samples} interacciones reales")
        historial = cargar_historial()
        if not historial:
            return {"error": "No hay historial para evaluar."}
        
        samples = []
        for r in historial[-max_samples:]:
            samples.append({
                "question": r["pregunta"],
                "answer": strip_html(r["respuesta"]),
                "contexts": [strip_html(str(f)) for f in r.get("fragmentos", [])] or ["Sin contexto"]
            })
    else:
        log_sequence("RAGAS Benchmark", f"Ejecutando Benchmark sobre {len(samples)} muestras")

    dataset = Dataset.from_list(samples)
    llm, embeddings = get_ragas_judges()
    
    # Definir métricas dinámicamente
    metrics_to_use = [faithfulness, answer_relevancy]
    if "ground_truth" in dataset.column_names:
        metrics_to_use.extend([context_precision, context_recall])
        log_info("RAGAS", "Modo Benchmark detectado: Habilitando Context Precision/Recall")
    else:
        log_info("RAGAS", "Modo Auditoría detectado: Usando fidelidad y relevancia únicamente")

    # Aplicar configuración bilingüe 🔥
    configure_bilingual_metrics(metrics_to_use)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=llm,
            embeddings=embeddings
        )
        
        df_results = result.to_pandas()
        # Extraer promedios de las métricas (excluyendo columnas de texto)
        numeric_scores = df_results.select_dtypes(include=['number']).mean().to_dict()
        scores = {k: float(v) for k, v in numeric_scores.items()}
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = DIR_OUTPUTS / f"ragas_report_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "scores": scores,
            "total_samples": len(samples),
            "is_benchmark": "ground_truth" in dataset.column_names
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # Actualizar último resultado para el Dashboard
        with open(DIR_OUTPUTS / "ragas_ultimo.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return report
    except Exception as e:
        log_error("Error en ejecución de RAGAS", e)
        return {"error": str(e)}

def cargar_ultima_evaluacion() -> dict:
    """Carga la última evaluación guardada o devuelve una por defecto para la UI."""
    ruta = DIR_OUTPUTS / "ragas_ultimo.json"
    if ruta.exists():
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    
    return {
        "scores": {"faithfulness": 0.0, "answer_relevancy": 0.0},
        "timestamp": "Sin datos",
        "total_samples": 0
    }

def eval_single_response(question: str, answer: str, contexts: list) -> dict:
    """Evaluación rápida de una respuesta individual (RAGAS Lite)."""
    if not question or not answer:
        return {"status": "error", "message": "Datos incompletos"}
    
    dataset = Dataset.from_list([{
        "question": question,
        "answer": strip_html(answer),
        "contexts": [strip_html(str(c)) for c in contexts] or ["Sin contexto"]
    }])
    llm, embeddings = get_ragas_judges()
    
    # Aplicar configuración bilingüe 🔥
    m_list = [faithfulness, answer_relevancy]
    configure_bilingual_metrics(m_list)
    
    try:
        result = evaluate(dataset=dataset, metrics=m_list, llm=llm, embeddings=embeddings)
        df = result.to_pandas()
        res_dict = df.select_dtypes(include=['number']).iloc[0].to_dict()
        
        # LOG DE DEPURACIÓN EN ARCHIVO (Para ver por qué da 0%)
        debug_path = DIR_LOGS / "ragas_debug.log"
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- {datetime.now().isoformat()} ---\n")
            f.write(f"Q: {question}\n")
            f.write(f"A: {strip_html(answer)}\n")
            f.write(f"Metrics: {res_dict}\n")
            f.write(f"Full Row: {df.iloc[0].to_dict()}\n")

        log_debug("RAGAS", f"Métricas crudas: {res_dict}")
        return {k: float(v) for k, v in res_dict.items()}
    except Exception as e:
        log_error(f"RAGAS Lite falló críticamente", e)
        # Intentar diagnóstico manual de por qué falló
        if not contexts or contexts == ["Sin contexto"]:
            log_warn("RAGAS falló: No hay contextos válidos para evaluar.")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}

if __name__ == "__main__":
    # Test rápido de registro
    registrar_consulta("¿Hola?", "¡Hola! Soy LegalGuard.", ["Trozo 1"], "doc.pdf", 0.95)
    print("Estadísticas:\n", calcular_stats_historial(cargar_historial()))
