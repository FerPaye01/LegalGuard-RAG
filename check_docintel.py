import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentModelAdministrationClient

load_dotenv()

def run_test():
    print("--- 📄 Probando Document Intelligence (Admin Mode) ---")
    try:
        admin_client = DocumentModelAdministrationClient(
            endpoint=os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_FORM_RECOGNIZER_KEY"))
        )
        details = admin_client.get_resource_details()
        print(f"✅ CONECTADO EXITOSAMENTE")
        print(f"📈 Límite de modelos: {details.custom_document_models.limit}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    run_test()