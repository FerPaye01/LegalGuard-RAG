import os
import json
import datetime
from pathlib import Path
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from dotenv import load_dotenv
from src.utils.logger import log_info, log_sequence, log_warn, log_error

load_dotenv()

# 1. Configuración de Content Safety (El Portero)
endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
safety_client = None
if endpoint and key:
    try:
        safety_client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
        log_info("Security", "Azure AI Content Safety conectado.")
    except Exception as e:
        log_error("Security", f"Error al conectar Content Safety: {e}")

# 2. Configuración de Presidio Local (El Cirujano) - FORZAR SOLO ESPAÑOL
# Nota: Requiere python -m spacy download es_core_news_lg
try:
    # Bloqueamos la descarga automática de inglés configurando el motor manualmente
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "es", "model_name": "es_core_news_lg"}],
    }
    # En la versión instalada se usa NlpEngineProvider
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
    anonymizer = AnonymizerEngine()
    log_info("Privacy", "Microsoft Presidio (NLP Local en ESPAÑOL) inicializado.")
except Exception as e:
    log_warn("Privacy", f"Falla al iniciar Presidio. ¿Bajaste el modelo es_core_news_lg? {e}")
    analyzer = None
    anonymizer = None

class GovernanceManager:
    """
    Gestor unificado de seguridad, privacidad y auditoría para LegalGuard.
    """
    def __init__(self, log_path: str = "outputs/governance/audit_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def check_content_safety(self, text: str):
        """Verifica si el texto es tóxico usando Azure AI Content Safety."""
        if not safety_client:
            return True, None
            
        request = AnalyzeTextOptions(text=text)
        try:
            response = safety_client.analyze_text(request)
            for category in response.categories_analysis:
                if category.severity > 2: # Umbral de severidad (0-7)
                    return False, f"Contenido bloqueado: {category.category}"
            return True, None
        except Exception as e:
            log_error("Security", f"Error en Content Safety: {e}")
            return True, None # Fallback permisivo para la demo en caso de caída de API

    def anonymize_legal_data(self, text: str):
        """Oculta PII (DNI, Personas, Teléfonos) localmente antes de mostrar al usuario."""
        if not analyzer or not anonymizer:
            return text
            
        # Analizamos en idioma español (es)
        results = analyzer.analyze(text=text, language='es', 
                                   entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])
        
        # Enmascaramos con reemplazos genéricos
        anonymized_result = anonymizer.anonymize(
            text=text, 
            analyzer_results=results
        )
        return anonymized_result.text

    def gatekeeper(self, text: str, is_input=True):
        """
        Función unificada de seguridad y filtrado.
        is_input=True: Valida toxicidad en la pregunta del usuario.
        is_input=False: Valida toxicidad y anonimiza la respuesta de la IA.
        """
        # 1. ¿Es seguro?
        is_safe, reason = self.check_content_safety(text)
        if not is_safe:
            return reason, False

        # 2. Anonimizar (solo en el output final para privacidad)
        if not is_input:
            clean_text = self.anonymize_legal_data(text)
            return clean_text, True
        
        return text, True

    def log_interaction(self, query: str, answer: str, documents: list, metadata: dict = None):
        """
        Registra cada interacción en el log inmutable de auditoría.
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "context_used": [
                {
                    "file": doc.get("source_file"),
                    "score": doc.get("score"),
                    "content": doc.get("content") # CRÍTICO para RAGAS
                } for doc in documents
            ],
            "metadata": metadata or {}
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log_error("Governance", f"Error al guardar log: {e}")

if __name__ == "__main__":
    # Test rápido de integración
    gov = GovernanceManager()
    output, safe = gov.gatekeeper("El DNI de Juan es 12345678A y su teléfono es 999888777", is_input=False)
    print(f"Resultado Seguro? {safe} | Texto: {output}")
