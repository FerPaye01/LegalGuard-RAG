from src.agent import LegalGuardAgent

class LegalGuardRAG:
    """
    Wrapper de compatibilidad para la Tarjeta de Ingesta.
    Centraliza las consultas al Agente Reactivo de LangGraph.
    """
    def __init__(self):
        self.agent = LegalGuardAgent()

    def query(self, question: str):
        """
        Lanza una consulta al motor RAG orquestado.
        Retorna: {
            "answer": str,
            "source": str,
            "confidence": float,
            "documents": list
        }
        """
        # Ejecutamos el grafo
        result = self.agent.run(question)
        
        # Formateamos para compatibilidad con la tarjeta
        return {
            "answer": result["answer"],
            "documents": result["documents"],
            "source": result["documents"][0]["source_file"] if result["documents"] else "N/A"
        }

if __name__ == "__main__":
    # Test rápido de integración
    rag = LegalGuardRAG()
    res = rag.query("¿Cuál es el periodo de terminación?")
    print(f"Respuesta: {res['answer']}")
    print(f"Fuente: {res['source']}")
