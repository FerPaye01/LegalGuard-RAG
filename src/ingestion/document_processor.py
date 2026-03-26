import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

# 1. Clientes para OPERAR (Los que ya te funcionan)
from openai import AzureOpenAI

# 2. Clientes para ADMINISTRAR (Los que fallaban)
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

if __name__ == "__main__":
    print("🚀 VALIDANDO INFRAESTRUCTURA RESTANTE\n")
    test_ai_search_fixed()
    test_doc_intel_fixed()