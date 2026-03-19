"""
Módulo para la configuración de Analyzer y AnonymizerEngine de Presidio (encriptación AES)
"""
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from presidio_anonymizer.entities import OperatorConfig

from src.config.settings import get_settings
from src.utils.logger import log_info, log_sequence, log_warn


# Entidades PII a anonimizar en contratos legales
PII_ENTITIES = ["PERSON", "LOCATION", "ORGANIZATION", "ID", "EMAIL_ADDRESS", "PHONE_NUMBER"]


def _build_analyzer() -> AnalyzerEngine:
    # Inicializa el motor de análisis con reconocedores predefinidos
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    return AnalyzerEngine(registry=registry, supported_languages=["en", "es"])


def anonymize_text(text: str) -> tuple[str, list]:
    # Detecta y encripta todas las entidades PII del texto para vectorización segura
    settings = get_settings()
    log_sequence("presidio: Analizando PII en texto", f"{len(text)} caracteres")

    analyzer = _build_analyzer()
    anonymizer = AnonymizerEngine()
    crypto_key = settings.presidio_encryption_key

    results = analyzer.analyze(text=text, language="en", entities=PII_ENTITIES)

    if not results:
        log_info("presidio: No se detectaron entidades PII")
        return text, []

    log_info(f"presidio: {len(results)} entidades PII detectadas")

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "DEFAULT": OperatorConfig("encrypt", {"key": crypto_key}),
        }
    )
    return anonymized.text, anonymized.items


def deanonymize_text(anonymized_text: str, anonymized_items: list) -> str:
    # Revierte la encriptación AES para mostrar datos originales al usuario final
    settings = get_settings()
    log_sequence("presidio: Deanonimizando respuesta del LLM", "")

    if not anonymized_items:
        log_warn("presidio: No hay items anonimizados para revertir")
        return anonymized_text

    deanonymizer = DeanonymizeEngine()
    crypto_key = settings.presidio_encryption_key

    result = deanonymizer.deanonymize(
        text=anonymized_text,
        entities=anonymized_items,
        operators={"DEFAULT": OperatorConfig("decrypt", {"key": crypto_key})}
    )
    log_info("presidio: Texto original recuperado correctamente")
    return result.text
