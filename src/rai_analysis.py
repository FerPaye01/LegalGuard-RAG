import os
import pandas as pd
from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIAnalyzer
from src.utils.logger import log_info, log_error

def setup_rai_dashboard(training_data, target_column):
    """
    Configura y lanza el dashboard de Responsible AI Toolbox para LegalGuard.
    """
    try:
        log_info("RAI", "Cargando datos para análisis de IA Responsable...")
        # (Este es un ejemplo de cómo se integraría con el Router de LegalGuard)
        # analyzer = RAIAnalyzer(model, training_data, test_data, target_column, "classification")
        # analyzer.compute()
        
        # ResponsibleAIDashboard(analyzer)
        log_info("RAI", "Dashboard de RAI configurado (Ejecutar en local interactivo).")
        
    except Exception as e:
        log_error("RAI", f"Error al configurar RAI Dashboard: {e}")

if __name__ == "__main__":
    # Generamos un dataset sintético para demostración de arquitectura
    data = {
        "question": [
            "¿Cuál es la fecha de terminación?", 
            "Dime un chiste", 
            "Tengo una duda sobre un NDA", 
            "¿Qué clima hace?"
        ],
        "intent": ["legal_search", "general_chat", "legal_search", "general_chat"]
    }
    df = pd.DataFrame(data)
    
    print("RAI Toolbox está listo para analizar el sesgo en el Router de LegalGuard.")
    # setup_rai_dashboard(df, "intent")
