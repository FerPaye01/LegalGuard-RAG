import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

load_dotenv()

def run_test():
    print("--- 🔍 Probando Azure AI Search (Admin Mode) ---")
    try:
        admin_client = SearchIndexClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )
        stats = admin_client.get_service_statistics()
        print(f"✅ CONECTADO EXITOSAMENTE")
        print(f"📊 Espacio usado: {stats['counters']['storage_size_counter']['usage']} bytes")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    run_test()