import os
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from src.utils.logger import log_info, log_error

class ContentSafetyManager:
    """
    Gestor de seguridad de contenido usando Azure AI Content Safety.
    Filtra categorías de Odio, Violencia, Autolesión y Contenido Sexual.
    """
    def __init__(self):
        endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        
        if not endpoint or not key:
            log_error("Security", "Credenciales de Azure Content Safety no configuradas en el entorno.")
            self.client = None
        else:
            self.client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    def analyze_text(self, text: str) -> bool:
        """
        Analiza un texto y retorna True si es SEGURO, False si debe ser bloqueado.
        """
        if not self.client:
            return True # Fallback por defecto si no hay cliente (permisivo)

        request = AnalyzeTextOptions(text=text)
        try:
            response = self.client.analyze_text(request)
            
            # Revisar severidad en cada categoría (0=Safe, 2, 4, 6=Harmful)
            for category_analysis in response.categories_analysis:
                if category_analysis.severity > 0:
                    log_info("Security", f"Contenido BLOQUEADO: Categoría {category_analysis.category} | Severidad {category_analysis.severity}")
                    return False
            
            return True
        except Exception as e:
            log_error("Security", f"Error en la llamada a Azure Content Safety: {e}")
            return True # En caso de error técnico, permitimos (o podrías bloquear por seguridad extrema)

if __name__ == "__main__":
    # Test rápido de seguridad (Simulado: Si no hay credenciales fallará graciosamente)
    from dotenv import load_dotenv
    load_dotenv()
    
    safety = ContentSafetyManager()
    print(f"¿Es seguro 'Hola mundo'?: {safety.analyze_text('Hola mundo')}")
