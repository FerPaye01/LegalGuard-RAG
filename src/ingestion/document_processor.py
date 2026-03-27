import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# 2. Clientes para ADMINISTRAR (Smoke tests)
from azure.search.documents.indexes import SearchIndexClient
from azure.ai.formrecognizer import DocumentModelAdministrationClient

load_dotenv()

def test_ai_search_fixed():
    print("--- 🔍 Probando Azure AI Search (Modo Admin) ---")
    try:
        # CAMBIO CLAVE: Usamos SearchIndexClient en lugar de SearchClient
        admin_client = SearchIndexClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )
        stats = admin_client.get_service_statistics()
        print(f"✅ AI Search Conectado. Almacenamiento: {stats['counters']['storage_size_counter']['usage']} bytes\n")
    except Exception as e:
        print(f"❌ Error en AI Search: {e}\n")

def test_doc_intel_fixed():
    print("--- 📄 Probando Document Intelligence (Modo Admin) ---")
    try:
        # CAMBIO CLAVE: Usamos DocumentModelAdministrationClient
        admin_client = DocumentModelAdministrationClient(
            endpoint=os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_FORM_RECOGNIZER_KEY"))
        )
        details = admin_client.get_resource_details()
        print(f"✅ Doc Intelligence Conectado. Límite de modelos: {details.custom_document_models.limit}\n")
    except Exception as e:
        print(f"❌ Error en Document Intelligence: {e}\n")

def extract_document_hybrid(file_path: str) -> str:
    """
    Analiza un documento PDF sucio usando el modelo 'prebuilt-layout' y fuerza una
    salida en formato Markdown puro PERO resguardando las tablas complejas bajo
    etiquetas HTML `<table>` nativas. Ideal para celdas combinadas (colspan).
    """
    print(f"🧠 document-intelligence: Solicitando prebuilt-layout (Híbrido) para {file_path}")
    
    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
    
    if not endpoint or not key:
        print("❌ Error: Faltan credenciales de Form Recognizer (.env)")
        return ""

    # Usamos el cliente v4 (DocumentIntelligenceClient) para habilitar output_content_format="markdown"
    client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # Iniciar el análisis exigiendo Markdown + Tablas HTML embebidas
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            body=pdf_bytes,
            output_content_format="markdown",
        )

        # Esperar el procesamiento asíncrono
        result = poller.result()

        if not result.content:
            print(f"⚠️ document-intelligence: Documento vacío o inprocesable: {file_path}")
            return ""

        print(f"✅ Extracción Híbrida completada exitosamente para {file_path}. Listo para el LLM.\n")
        return result.content
        
    except Exception as e:
        print(f"❌ document-intelligence: Fallo en Azure. Endpoint inactivo o PDF corrupto: {e}\n")
        return ""

if __name__ == "__main__":
    print("🚀 VALIDANDO INFRAESTRUCTURA DE EXTRACCIÓN\n")
    test_ai_search_fixed()
    test_doc_intel_fixed()