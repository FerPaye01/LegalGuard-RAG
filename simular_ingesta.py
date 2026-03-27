import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Cliente de OpenAI para generar el vector
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Nuestro documento de prueba
doc_simulado = {
    "id": "CONTRATO-2026-001",
    "titulo": "Contrato de Locación de Servicios - Oscar Paye",
    "contenido": "El locador se compromete a realizar el mantenimiento preventivo de los equipos informáticos durante el periodo fiscal 2026."
}

def generar_vector(texto):
    print(f"🧠 Generando vector para: {texto[:30]}...")
    response = client.embeddings.create(
        input=texto,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    return response.data[0].embedding

# Simulamos la "Vectrización"
vector_contenido = generar_vector(doc_simulado["contenido"])

# Así quedaría el objeto listo para subir a Azure AI Search
payload_busqueda = {
    "id": doc_simulado["id"],
    "titulo": doc_simulado["titulo"],
    "contenido": doc_simulado["contenido"],
    "contenido_vector": vector_contenido, # <--- El vector de 1536 dimensiones
    "metadata": "MANTENIMIENTO-2026"
}

print("\n✅ Objeto preparado para Azure AI Search.")
print(f"Dimensiones del vector: {len(payload_busqueda['contenido_vector'])}")
print(f"ID del documento: {payload_busqueda['id']}")