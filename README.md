# ⚖️ LegalGuard RAG

> **Asistente inteligente de revisión de contratos con gobernanza**
> Sistema RAG adaptativo que simplifica la revisión legal y reduce la carga cognitiva en equipos regulados mediante IA responsable sobre Azure.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![Azure AI Search](https://img.shields.io/badge/Azure-AI%20Search-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/products/ai-services/ai-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Innovation Challenge](https://img.shields.io/badge/Microsoft-Innovation%20Challenge%202026-purple)](https://innovationstudio.microsoft.com)

---

## 📋 Tabla de contenidos

- [Sobre el proyecto](#-sobre-el-proyecto)
- [Características principales](#-características-principales)
- [Arquitectura Azure](#-arquitectura-azure)
- [Servicios Azure utilizados](#-servicios-azure-utilizados)
- [Datasets](#-datasets)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Evaluación y métricas](#-evaluación-y-métricas)
- [IA Responsable](#-ia-responsable)
- [Equipo](#-equipo)
- [Desafío del hackathon](#-desafío-del-hackathon)

---

## 🎯 Sobre el proyecto

Los equipos legales, de compliance, finanzas y salud pierden horas revisando contratos manualmente para identificar cláusulas de riesgo. **LegalGuard RAG** es un sistema de Generación Aumentada por Recuperación (RAG) con gobernanza que responde preguntas complejas sobre contratos, **citando siempre la fuente exacta** y **sin riesgo de alucinaciones**.

### El problema que resuelve

- ⏱️ Un abogado tarda 3-6 horas en revisar un contrato complejo
- ❌ Los LLM generativos "inventan" cláusulas que no existen (alucinaciones)
- 📄 Los equipos regulados necesitan trazabilidad completa de cada decisión
- 🔍 Identificar cláusulas de riesgo en documentos de 50+ páginas es propenso a error humano

### Nuestra solución

LegalGuard RAG escanea automáticamente contratos legales, identifica los 41 tipos de cláusulas críticas (según el dataset CUAD anotado por abogados), genera un **score de riesgo 0-100** y responde preguntas en lenguaje natural citando el párrafo exacto de origen.

---

## ✨ Características principales

### 💬 Asistente de preguntas sobre contratos
- Responde preguntas en lenguaje natural sobre cualquier contrato
- **Cada respuesta cita el fragmento exacto** del documento de origen
- Indicador de confianza: si la información no está en el documento, el sistema lo dice
- Historial completo y auditable de consultas

### 🔍 Risk Scanner (innovación principal)
- Escaneo automático de contratos completos
- Detecta los **41 tipos de cláusulas críticas** (CUAD) presentes y ausentes
- Genera un **score de riesgo 0-100** con justificación
- Exporta informe de análisis en JSON/PDF

### 🛡️ Gobernanza y IA Responsable
- Filtro de alucinaciones: umbral de confianza configurable
- Azure AI Content Safety integrado (protección de inputs y outputs)
- Dashboard de auditoría con Responsible AI Toolbox de Microsoft
- Trazabilidad completa: pregunta → fragmentos recuperados → respuesta → fuente

### 🌐 Multi-dominio
- Dominio legal: contratos comerciales, NDAs, acuerdos de compliance
- Dominio salud: procedimientos operativos estándar (SOPs) de la OMS

---

## 🏗️ Arquitectura Azure

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO                                 │
│              "¿Cuál es la cláusula de terminación?"             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Azure App Service (Streamlit)                      │
│         Interfaz web · Risk Scanner · Dashboard                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│  Azure AI   │  │  Azure OpenAI│  │  Azure AI        │
│  Content    │  │  GPT-4o      │  │  Content Safety  │
│  Search     │  │  Embeddings  │  │  (filtro I/O)    │
│  (vectorial)│  │  ada-002     │  └──────────────────┘
└─────────────┘  └──────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Azure Blob Storage                           │
│         Contratos PDF · NDAs · SOPs · Logs JSON                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────┐          ┌──────────────────────┐
│  Azure Document │          │  Azure Monitor +     │
│  Intelligence   │          │  Application Insights│
│  (extrae PDFs)  │          │  (trazabilidad)      │
└─────────────────┘          └──────────────────────┘
```

---

## ☁️ Servicios Azure utilizados

| Servicio | Propósito | Tier |
|----------|-----------|------|
| **Azure OpenAI Service** | LLM (GPT-4o) + embeddings (ada-002) | Standard |
| **Azure AI Search** | Base vectorial, búsqueda semántica híbrida | Basic |
| **Azure Document Intelligence** | Extracción de texto de contratos PDF | Free (500 pág/mes) |
| **Azure AI Content Safety** | Filtro de inputs/outputs peligrosos | Free tier |
| **Azure Blob Storage** | Almacén de documentos y logs | LRS Standard |
| **Azure App Service** | Deploy de la aplicación web | F1 Free |
| **Azure Monitor + App Insights** | Trazabilidad y observabilidad | Free tier |
| **Azure AI Studio** | Orquestación y evaluación del pipeline | — |

---

## 📊 Datasets

| Dataset | Uso | Registros |
|---------|-----|-----------|
| [CUAD (Atticus Project)](https://huggingface.co/datasets/theatticusproject/cuad) | Base de conocimiento principal + evaluación | 510 contratos · 41 tipos de cláusulas |
| [Legal Contract Q&A (Strova)](https://huggingface.co/datasets/strova-ai/legal_contract_dataset) | Demo y validación rápida | Sintético JSONL |
| [ContractNLI (Stanford)](https://stanfordnlp.github.io/contract-nli/) | Métricas de implicación/contradicción | 607 NDAs |
| [WHO SOP PDF](https://platform.who.int/docs/default-source/mca-documents/) | Multi-dominio salud | 1 documento |

---

## 🚀 Instalación

### Prerrequisitos
- Python 3.10+
- Cuenta Azure (Azure for Students recomendado: $100 crédito gratuito)
- Git

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/legalguard-rag.git
cd legalguard-rag
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Copia el archivo de ejemplo y completa con tus credenciales Azure:

```bash
cp .env.example .env
```

Edita `.env` con tus valores:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://TU-RECURSO.openai.azure.com/
AZURE_OPENAI_API_KEY=tu_api_key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://TU-RECURSO.search.windows.net
AZURE_SEARCH_API_KEY=tu_api_key
AZURE_SEARCH_INDEX_NAME=contratos-index

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_STORAGE_CONTAINER=contratos

# Azure Document Intelligence
AZURE_FORM_RECOGNIZER_ENDPOINT=https://TU-RECURSO.cognitiveservices.azure.com/
AZURE_FORM_RECOGNIZER_KEY=tu_api_key

# Azure Content Safety
AZURE_CONTENT_SAFETY_ENDPOINT=https://TU-RECURSO.cognitiveservices.azure.com/
AZURE_CONTENT_SAFETY_KEY=tu_api_key
```

### 4. Cargar los documentos

```bash
# Descarga y procesa el dataset CUAD
python src/ingestion.py --dataset cuad --sample 50

# Procesa el PDF de la OMS
python src/ingestion.py --file data/who_sop.pdf --domain health
```

### 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`

---

## 💻 Uso

### Consulta sobre un contrato

```python
from src.rag_engine import LegalGuardRAG

rag = LegalGuardRAG()

# Pregunta sobre un contrato
result = rag.query(
    question="¿Cuál es el periodo de terminación del contrato?",
    domain="legal"
)

print(result["answer"])          # Respuesta generada
print(result["source"])          # Nombre del contrato y párrafo
print(result["confidence"])      # Score de confianza 0-1
print(result["fragment"])        # Fragmento exacto del documento
```

### Risk Scanner

```python
from src.risk_scanner import RiskScanner

scanner = RiskScanner()

# Escanear un contrato completo
report = scanner.scan("data/contracts/sample_nda.pdf")

print(f"Risk Score: {report['risk_score']}/100")
print(f"Cláusulas detectadas: {report['clauses_found']}")
print(f"Cláusulas faltantes: {report['clauses_missing']}")
```

---

## 📈 Evaluación y métricas

El sistema se evalúa usando el framework **RAGAS** con el dataset CUAD como ground truth:

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **Faithfulness** | Respuestas respaldadas por los documentos | > 0.85 |
| **Answer Relevancy** | Relevancia de la respuesta a la pregunta | > 0.80 |
| **Context Precision** | Precisión de los fragmentos recuperados | > 0.75 |
| **Context Recall** | Cobertura de información relevante | > 0.70 |

Para ejecutar la evaluación:

```bash
python src/metrics.py --dataset cuad --n_samples 50
```

---

## 🤝 IA Responsable

LegalGuard RAG implementa los principios de IA Responsable de Microsoft:

- **Confiabilidad**: umbral de confianza configurable; el sistema admite cuando no tiene información suficiente
- **Seguridad**: Azure AI Content Safety filtra inputs y outputs en tiempo real
- **Privacidad**: los documentos no salen del tenant Azure del usuario
- **Equidad**: evaluación con Responsible AI Toolbox para detectar sesgos en respuestas
- **Transparencia**: cada respuesta muestra su fuente, score de confianza y fragmento original
- **Trazabilidad**: historial completo de consultas disponible en Azure Monitor

---

## 👥 Equipo

| Nombre | Rol | Responsabilidad |
|--------|-----|----------------|
| **Estefany Paola Mamani Gutierrez** | Backend / RAG Engineer | Pipeline RAG, Azure AI Search, embeddings, Risk Scanner, RAGAS |
| **Oscar Fernando Paye Cahui** | Frontend / Integration | Interfaz Streamlit, deploy Azure, presentación, video demo |

**Universidad**: [Tu universidad]
**Hackathon**: Microsoft Innovation Challenge March 2026
**Desafío elegido**: Asistente para la reducción de la carga cognitiva

---

## 🏆 Desafío del hackathon

Este proyecto participa en el **Microsoft Innovation Challenge March 2026**, resolviendo el desafío:

> *"Asistente para la reducción de la carga cognitiva — Un sistema de agentes adaptativo que simplifica el trabajo y el aprendizaje para usuarios neurodiversos."*

### Criterios de evaluación

| Criterio | Peso | Nuestra implementación |
|----------|------|----------------------|
| Rendimiento | 25% | Respuestas < 3s · RAGAS score > 0.80 |
| Innovación | 25% | Risk Scanner con score 0-100 · Multi-dominio |
| Amplitud Azure | 25% | 7 servicios Azure integrados |
| IA Responsable | 25% | Content Safety + RAI Toolbox + trazabilidad |

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

## 🙏 Reconocimientos

- [The Atticus Project](https://www.atticusprojectai.org/) por el dataset CUAD
- [Stanford NLP](https://stanfordnlp.github.io/contract-nli/) por ContractNLI
- [Microsoft Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)
- [RAGAS](https://docs.ragas.io/) por el framework de evaluación RAG

---

<p align="center">
  Construido con ❤️ por el equipo LegalGuard · Innovation Challenge March 2026
</p>
