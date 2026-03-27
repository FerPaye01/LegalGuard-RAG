import os
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- 1. ESTRUCTURAS DE DATOS (PYDANTIC) ---
class ClauseResult(BaseModel):
    clause_name: str = Field(description="Nombre de la cláusula analizada (ej. 'Terminación')")
    is_present: bool = Field(description="¿Se encontró esta cláusula explícitamente en el contrato?")
    risk_weight: int = Field(description="Peso de riesgo original (1=Bajo, 2=Medio, 3=Crítico)")
    excerpt: Optional[str] = Field(description="Si se encontró, extrae un breve fragmento literal que lo demuestre.")
    comment: Optional[str] = Field(description="Breve comentario legal sobre el nivel de riesgo de esta cláusula o su ausencia.")

class RiskReport(BaseModel):
    total_score: float = Field(description="Score de riesgo calculado de 0 a 100 (100 = Riesgo Máximo, 0 = Sin Riesgo).")
    clauses: List[ClauseResult] = Field(description="Análisis detallado de cada una de las 41 cláusulas evaluadas.")
    missing_critical: List[str] = Field(description="Lista de nombres de cláusulas críticas (peso 3) que faltan.")

# --- 2. CONFIGURACIÓN DEL CLIENTE ---
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

MINI_DEPLOYMENT = os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-4o-mini")

# --- 3. DICCIONARIO MAESTRO (41 CLÁUSULAS CUAD) ---
CLAUSES_MAP = [
    {"name": "Nombre del Documento", "weight": 1},
    {"name": "Partes", "weight": 2},
    {"name": "Fecha del Acuerdo", "weight": 1},
    {"name": "Fecha de Vigencia", "weight": 2},
    {"name": "Fecha de Expiración", "weight": 2},
    {"name": "Término de Renovación", "weight": 2},
    {"name": "Periodo de Notificación para Terminar Renovación", "weight": 2},
    {"name": "Ley Aplicable y Jurisdicción", "weight": 3},
    {"name": "Resolución de Disputas", "weight": 3},
    {"name": "Arbitraje", "weight": 3},
    {"name": "Sede / Tribunal", "weight": 3},
    {"name": "Concesión de Licencia", "weight": 2},
    {"name": "Licencia Intransferible", "weight": 2},
    {"name": "Exclusividad", "weight": 3},
    {"name": "No Solicitud de Clientes", "weight": 2},
    {"name": "Excepción a la Restricción Competitiva", "weight": 2},
    {"name": "No Desprestigio", "weight": 1},
    {"name": "Terminación por Conveniencia", "weight": 3},
    {"name": "Derecho de Preferencia / ROFR / ROFO", "weight": 2},
    {"name": "Cambio de Control", "weight": 3},
    {"name": "Anti-Cesión", "weight": 2},
    {"name": "Reparto de Ingresos / Beneficios", "weight": 2},
    {"name": "Restricción de Precio", "weight": 2},
    {"name": "Restricción de Volumen", "weight": 2},
    {"name": "Asignación de Propiedad Intelectual", "weight": 3},
    {"name": "Propiedad Intelectual Conjunta", "weight": 2},
    {"name": "Pacto de No Demandar", "weight": 2},
    {"name": "Indemnización", "weight": 3},
    {"name": "Daños y Perjuicios Liquidados (Penalidades)", "weight": 3},
    {"name": "Límite de Responsabilidad", "weight": 3},
    {"name": "Excepciones al Límite de Responsabilidad", "weight": 3},
    {"name": "Duración de la Garantía", "weight": 2},
    {"name": "Requisito de Seguro", "weight": 2},
    {"name": "Derechos de Auditoría", "weight": 2},
    {"name": "Retención de Registros", "weight": 1},
    {"name": "Supervivencia de Cláusulas (Survival)", "weight": 1},
    {"name": "Periodo de Notificación", "weight": 1},
    {"name": "Fuerza Mayor", "weight": 2},
    {"name": "Confidencialidad General / NDA", "weight": 3},
    {"name": "Límite de Publicación", "weight": 2},
    {"name": "Parte Prevaleciente (Costas Judiciales)", "weight": 1}
]

# --- 4. MOTOR DE ESCANEO ---
def scan_contract(markdown_text: str) -> RiskReport:
    """
    Analiza un contrato completo en busca de las 41 cláusulas de CUAD usando Structured Outputs.
    Maneja hasta 128k tokens eficientemente con gpt-4o-mini.
    """
    print(f"🚀 Iniciando escaneo profundo con {MINI_DEPLOYMENT}...")
    
    # Preparamos las instrucciones para el LLM
    clauses_instruction = "\n".join([f"- {c['name']} (Peso de riesgo: {c['weight']})" for c in CLAUSES_MAP])
    
    system_prompt = f"""
    Eres un auditor legal implacable. Tu objetivo es escanear detenidamente el contrato proporcionado y verificar la presencia explícita de las siguientes 41 cláusulas estándar del framework CUAD:
    
    {clauses_instruction}
    
    Reglas Arquitectónicas:
    1. Para cada una de las 41 cláusulas, determina 'is_present'.
    2. Si 'is_present' es true, DEBES extraer un fragmento crítico textual en 'excerpt' que justifique el hallazgo.
    3. Si falta una cláusula de Peso 3 (Crítico), agrégala obligatoriamente a la lista general 'missing_critical'.
    4. El 'total_score' debe reflejar algorítmicamente el riesgo: 
       - 100 = Peligro Máximo (faltan las cláusulas de pago, penalidades, ley, etc.)
       - 0 = Contrato Perfecto y blindado.
       - Las de Peso 3 impactan masivamente la penalización si no están.
       - Sé objetivo con el cálculo matemático del score antes de retornar.
    """

    # Structured Output parse for 100% JSON consistency
    try:
        response = client.beta.chat.completions.parse(
            model=MINI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                # Extraemos hasta los primeros 100k caracteres (suele ser suficiente para contratos de 50 pág)
                {"role": "user", "content": f"Audita este contrato y reporta hallazgos sobre las 41 cláusulas:\n\n{markdown_text[:100000]}"}
            ],
            response_format=RiskReport,
            temperature=0.0 # Temperatura cero para análisis de hechos legales (no creatividad)
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"Error crítico en Risk Scanner: {e}")
        raise e
