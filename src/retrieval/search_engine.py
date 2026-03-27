import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from src.utils.logger import log_info, log_sequence, log_warn, log_error

load_dotenv()

class AzureSearchHybridEngine:
    """
    Motor híbrido de Búsqueda RAG.
    Utiliza el paradigma de Reciprocal Rank Fusion (RRF).
    Calcula simultáneamente la similitud de coseno del Vector (HNSW) y
    el algoritmo BM25 de palabras clave para evitar los Falsos Positivos Vectoriales.
    """
    def __init__(self):
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "contratos-index")
        
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        self.oai_client = AzureOpenAI(
            api_key=self.openai_key,
            api_version=self.openai_version,
            azure_endpoint=self.openai_endpoint
        )
        
        self.search_client = SearchClient(
            endpoint=self.search_endpoint, 
            index_name=self.index_name, 
            credential=AzureKeyCredential(self.search_key)
        )

    def _get_embedding(self, text):
        response = self.oai_client.embeddings.create(input=text, model=self.embedding_deployment)
        return response.data[0].embedding
        
    def search_hybrid(self, query: str, top_k: int = 3):
        log_sequence("Ejecutando Búsqueda Híbrida (BM25 + HNSW RRF)", query)
        
        try:
            # 1. Transformar la pregunta del usuario a matemáticas geonómicas
            vector = self._get_embedding(query)
            vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=top_k, fields="content_vector")
            
            # 2. Inyectar la solicitud concurrente a Azure
            # Al enviar search_text y vector_queries en paralelo, Azure fusiona el score (RRF).
            results = self.search_client.search(
                search_text=query,                 # Componente Léxico (Mitiga IDs y Nombres Exactos)
                vector_queries=[vector_query],     # Componente Vectorial (HNSW)
                select=["id", "source_file", "content"],
                top=top_k
            )
            
            extracted_results = []
            for doc in results:
                score = doc.get("@search.score", 0)
                log_info(f"Chunk extraído validado en {doc['source_file']} con Score Híbrido: {score:.5f}")
                extracted_results.append({
                    "id": doc["id"],
                    "source_file": doc["source_file"],
                    "content": doc["content"],
                    "score": score
                })
                
            return extracted_results
            
        except Exception as e:
            log_error("Fallo durante la búsqueda en AI Search", e)
            return []

if __name__ == "__main__":
    # Test Unitario Directo para confirmar al Usuario
    test_query = "confidential information and disclosures"
    print(f"\n🔍 Probando motor con: '{test_query}'")
    
    engine = AzureSearchHybridEngine()
    resultados = engine.search_hybrid(test_query)
    
    print("\n==================================")
    print("      ✅ TOP RESULTADOS RRF      ")
    print("==================================")
    for r in resultados:
        print(f"📄 Origen: {r['source_file']}")
        print(f"🎯 Score: {r['score']:.4f}")
        print(f"📝 Extracto: {r['content'][:250]}...\n")
