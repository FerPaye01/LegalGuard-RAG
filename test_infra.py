import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient

# 1. Cargar configuración del .env
load_dotenv()

def test_azure_openai():
    print("--- Probando Azure OpenAI (Chat) ---")
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            messages=[{"role": "user", "content": "Hola, ¿estás listo para analizar leyes?"}],
            max_tokens=10
        )
        print(f"✅ OpenAI responde: {response.choices[0].message.content}\n")
    except Exception as e:
        print(f"❌ Error en OpenAI: {e}\n")

def test_embeddings():
    print("--- Probando Azure OpenAI (Embeddings) ---")
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        response = client.embeddings.create(
            input="prueba de vectores",
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        print(f"✅ Embeddings generados con éxito (Dim: {len(response.data[0].embedding)})\n")
    except Exception as e:
        print(f"❌ Error en Embeddings: {e}\n")

def test_ai_search():
    print("--- Probando Azure AI Search ---")
    try:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_API_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        client = SearchClient(endpoint, index_name, AzureKeyCredential(key))
        # Intentamos obtener estadísticas del índice (aunque esté vacío)
        stats = client.get_index_statistics()
        print(f"✅ Conexión con AI Search exitosa. Documentos actuales: {stats['document_count']}\n")
    except Exception as e:
        print(f"❌ Error en AI Search (Nota: Si el índice no existe aún, es normal): {e}\n")

def test_document_intelligence():
    print("--- Probando Document Intelligence ---")
    try:
        endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
        key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
        client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
        
        # Solo listamos los modelos disponibles para probar la llave
        models = client.get_resource_details()
        print(f"✅ Document Intelligence activo. Límite de modelos: {models.custom_document_models.limit}\n")
    except Exception as e:
        print(f"❌ Error en Document Intelligence: {e}\n")

def test_session_pool():
    print("--- Probando Container App Session Pool ---")
    try:
        pool_url = os.getenv("AZURE_CONTAINER_APP_SESSION_POOL")
        # Azure requiere una API Key o Token para el plano de datos usualmente, 
        # pero para el Smoke Test verificamos si el endpoint es alcanzable.
        response = requests.get(f"{pool_url}/health") 
        if response.status_code in [200, 401, 403]: # 401/403 significa que el endpoint existe pero pide auth
            print(f"✅ Session Pool alcanzable (Status: {response.status_code})\n")
        else:
            print(f"⚠️ Session Pool respondió con status: {response.status_code}\n")
    except Exception as e:
        print(f"❌ Error al contactar Session Pool: {e}\n")

if __name__ == "__main__":
    print("🚀 INICIANDO SMOKE TEST DE LEGALGUARD\n")
    test_azure_openai()
    test_embeddings()
    test_ai_search()
    test_document_intelligence()
    test_session_pool()
    print("🏁 PRUEBAS FINALIZADAS")