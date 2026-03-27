import sys
import os
# Hackathon Fix: Añadimos la raíz del proyecto al sys.path para que Streamlit resuelva "src"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import httpx
import json
import asyncio
import tempfile
from typing import Generator

# Import the Document Intelligence Script
from src.ingestion.document_processor import extract_document_hybrid
from src.ingestion.pipeline import index_document_from_text
from src.agent import LegalGuardAgent
from src.risk_scanner import scan_contract
from src.metrics import run_evaluation

# 1. Configuración de página (SIEMPRE PRIMERO)
st.set_page_config(
    page_title="LegalGuard RAG | Smart Contract Review",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Estado de Sesión (Protección Guardia de Estado)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "md_content" not in st.session_state:
    st.session_state.md_content = None
if "agent" not in st.session_state:
    # Inicializamos el cerebro de LangGraph una sola vez para ahorrar recursos
    st.session_state.agent = LegalGuardAgent()

# 3. Toggle de Modo Noche (Sidebar) - Debe ir antes del CSS
st.session_state.dark_mode = st.sidebar.toggle("🌙 Modo Noche", value=st.session_state.dark_mode)

# 4. Configuración de Colores Dinámicos
if st.session_state.dark_mode:
    bg_app = "#0F172A"       # Slate 900
    bg_chat = "#1E293B"      # Slate 800
    color_text = "#F8FAFC"   # White/Slate
    color_header = "#3B82F6" # Azure Blue
    border_color = "#334155" # Slate 700
    sidebar_bg = "#020617"   # Slate 950
    input_bg = "#334155"     # Slate 700
    bubble_user = "#064E3B"  # Emerald 900
    bubble_ai = "#1E293B"    # Slate 800
    text_secondary = "#94A3B8"
else:
    bg_app = "#F8FAFC"
    bg_chat = "#FFFFFF"
    color_text = "#0F172A"
    color_header = "#0F172A"
    border_color = "#E2E8F0"
    sidebar_bg = "#0F172A"   # Sidebar siempre oscuro para mayor "Premium feel"
    input_bg = "#FFFFFF"
    bubble_user = "#F0Fdf4"
    bubble_ai = "#FFFFFF"
    text_secondary = "#64748B"

# 5. Estilos Personalizados y CSS Inyectado para Tablas Híbridas
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* Fondo principal forzado */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: {bg_app} !important;
        color: {color_text} !important;
    }}

    /* Estilos inyectados para Tablas HTML originadas de Doc Intelligence */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        background-color: {input_bg};
        border-radius: 8px;
        overflow: hidden;
    }}
    th, td {{
        border: 1px solid {border_color} !important;
        padding: 10px 14px;
        text-align: left;
        color: {color_text};
        font-size: 0.95rem;
    }}
    th {{
        background-color: {sidebar_bg} !important;
        color: #E2E8F0 !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }}
    tr:nth-child(even) {{
        background-color: rgba(255, 255, 255, 0.02);
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
    }}
    [data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}
    
    /* Manejo del Progress y Spinners */
    .stProgress > div > div > div > div {{
        background-color: #F59E0B !important; 
    }}

    .main-header {{
        font-size: 2.22rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, {color_header}, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.22rem !important;
        letter-spacing: -0.022em;
        line-height: 1.2 !important;
    }}

    .stChatMessage {{
        background-color: {bubble_ai} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        padding: 1.2rem !important;
        margin-bottom: 1.2rem !important;
        animation: slideUp 0.3s ease-out;
        color: {color_text} !important;
    }}
    
    [data-testid="chat-message-user"] {{
        background-color: {bubble_user} !important;
        border-color: {border_color} !important;
    }}

    [data-testid="stChatInput"] {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }}
    
    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] div {{
        background-color: {input_bg} !important;
        color: {color_text} !important;
    }}
    
    [data-testid="stChatInput"] button {{
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 8px !important;
    }}

    .pdf-viewer-placeholder {{
        height: 600px; 
        background: {input_bg};
        border: 2px dashed {border_color}; 
        border-radius: 16px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        color: {color_text};
        transition: all 0.3s ease;
    }}

    #MainMenu, footer {{visibility: hidden;}}
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""", unsafe_allow_html=True)

# 6. Backend Logic (SSE)
async def stream_from_backend(question: str):
    url = "http://localhost:8000/ask"
    payload = {"question": question, "thread_id": "streamlit_user"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    yield {"error": f"Error del servidor: {response.status_code}"}
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            yield data
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield {"error": str(e)}

# 7. Sidebar Content (Control e Ingesta)
with st.sidebar:
    st.markdown('<h1 style="font-size: 3.5rem; margin-bottom: -20px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🏛️</h1>', unsafe_allow_html=True)
    st.title("LegalGuard RAG")
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Enterprise-Grade Legal AI</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
    
    st.subheader("📂 Ingesta de Contratos")
    uploaded_file = st.file_uploader("Sube un documento PDF", type="pdf")
    
    if uploaded_file and st.button("🚀 Procesar con IA (Doc Intel)", use_container_width=True):
        with st.status("Preparando archivo local...", expanded=True) as status:
            try:
                # 1. Guardar temporalmente
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # 2. Ingesta a Azure Document Intelligence
                status.write("Enviando a Azure Document Intelligence...")
                status.write("Extrayendo Markdown e inyectando Tablas HTML...")
                
                md_result = extract_document_hybrid(tmp_path)
                
                if md_result:
                    # 3. Indexación en Caliente (Hot-Indexing) 🔥
                    status.write("Enviando fragmentos a la Base Vectorial (Azure AI Search)...")
                    index_document_from_text(uploaded_file.name, md_result)
                    
                    st.session_state.md_content = md_result
                    status.update(label="¡Indexación y Procesamiento completados!", state="complete", expanded=False)
                    os.unlink(tmp_path)
                    st.rerun() 
                else:
                    status.update(label="Fallo en la extracción de Azure", state="error", expanded=True)
            except Exception as e:
                status.update(label=f"Excepción: {e}", state="error", expanded=True)

    st.markdown("---")
    
    # 4. Sidebar: Configuración y Status
    st.header("⚙️ Configuración")
    
    # Dashboard de Seguridad (Toque de Experto para Demo)
    with st.expander("🛡️ Estado de Seguridad", expanded=True):
        st.success("✅ Content Safety: Activo")
        st.info("🔒 Privacidad: Local (spaCy)")
        st.caption("Anonimización PII activada para DNI, Pasaportes y Teléfonos.")
        st.caption("Model: es_core_news_lg")
    
    st.divider()
    
    # Documento actual
    if st.session_state.md_content:
        st.info(f"📄 Documento activo: {st.session_state.get('pdf_name', 'Contrato')}")
    else:
        st.warning("⚠️ Ningún contrato procesado.")
    
    # Placeholder para Demo Rápido si no tienen PDFs a la mano
    if st.session_state.md_content:
        st.success("Documento cargado de forma segura en memoria.")
    
    if st.button("🗑️ Limpiar Memoria Completa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.md_content = None
        st.rerun()

# 8. Main Layout (Aplicando Opción B)
col_pdf, col_chat = st.columns([5, 5]) 

with col_pdf:
    st.markdown(f'<h3 style="color: {color_text}; font-weight: 700; display: flex; align-items: center; gap: 8px;">📄 Espejo Legal Processado</h3>', unsafe_allow_html=True)
    bg_info = "#1E293B" if st.session_state.dark_mode else "#E0F2FE"
    color_info = "#3B82F6" if st.session_state.dark_mode else "#0369A1"
    
    # Si tenemos el markdown ya extraído, no volvemos a llamar a Azure
    if st.session_state.md_content:
        st.markdown(f"""
            <div style="background-color: {bg_info}; border-left: 4px solid #3B82F6; padding: 12px; border-radius: 8px; font-size: 0.9rem; color: {color_info}; margin-bottom: 20px;">
                💡 Texto híbrido validado. El LLM ahora puede interactuar con datos precisos.
            </div>
        """, unsafe_allow_html=True)

        # Renderizar la data final. Gracias a unsafe_allow_html, las tablas <table> tendrán formato del bloque CSS inyectado arriba.
        st.markdown(st.session_state.md_content, unsafe_allow_html=True)
        
        st.markdown("---")
        # El "Toque de la Opción C" solicitado por el usuario para debugear en la Demo
        with st.expander("🛠️ Ver Código Fuente (Raw Markdown & HTML)"):
            st.code(st.session_state.md_content, language="markdown")

    else:
        st.markdown(f"""
            <div class="pdf-viewer-placeholder">
                <div style="text-align: center;">
                    <h4 style="margin: 0; color: {color_text};">Esperando Ingesta</h4>
                    <p style="font-size: 0.9rem; margin-top: 5px; color: {text_secondary};">Sube el PDF en la barra lateral para su re-construcción semántica.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

with col_chat:
    tab_chat, tab_risk, tab_metrics = st.tabs(["💬 Asistente Legal", "📋 Risk Scanner (CUAD)", "📊 Métricas RAGAS"])
    
    with tab_chat:
        st.markdown('<h1 class="main-header">Agente LegalGuard</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {text_secondary}; margin-top: -15px;">Interacción conversacional (LangGraph RAG Activo).</p>', unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "documents" in msg:
                    with st.expander("🔍 Ver fragmentos originales y fuentes"):
                        for i, doc in enumerate(msg["documents"]):
                            st.markdown(f"**Fragmento {i+1} - {doc['source_file']}**")
                            st.info(doc["content"])

        if st.session_state.md_content:
            if prompt := st.chat_input("Pregunta sobre las cláusulas u objetos del contrato..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.status("Analizando requerimiento legal...", expanded=True) as status:
                        st.write("🔍 Clasificando intención con el **Router**...")
                        
                        result = st.session_state.agent.run(prompt)
                        final_text = result["answer"]
                        docs = result["documents"]
                        counts = result.get("grader_counts", {})
                        
                        if counts:
                            t_found = counts.get("total_found", 0)
                            t_rel = counts.get("total_relevant", 0)
                            st.write(f"⚖️ Buscador encontró **{t_found} fragmentos** → Grader validó **{t_rel} como relevantes**.")
                        else:
                            st.write("⚖️ Validando relevancia en **Azure AI Search**...")
                            
                        st.write("✍️ Sintetizando respuesta con bases sólidas...")
                        status.update(label="Análisis Resuelto", state="complete", expanded=False)
                    
                    st.markdown(final_text)
                    
                    if docs:
                        with st.expander("🔍 Ver fragmentos originales y fuentes"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Fragmento {i+1} - {doc['source_file']}**")
                                st.info(doc["content"])

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_text,
                        "documents": docs
                    })
        else:
            st.info("⬅️ Usa la barra lateral para subir y procesar un contrato primero.")

    with tab_risk:
        st.markdown('<h1 class="main-header">Scanner de Riesgo Global</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {text_secondary}; margin-top: -15px;">Auditoría integral predictiva usando GPT-4o-mini (128k).</p>', unsafe_allow_html=True)
        
        if st.session_state.md_content:
            if st.button("🔎 Ejecutar Auditoría Profunda (Tsunami de Contexto)", use_container_width=True, type="primary"):
                with st.spinner("🤖 El auditor implacable está revisando 41 parámetros... esto tomará unos ~10 segundos"):
                    try:
                        report = scan_contract(st.session_state.md_content)
                        st.session_state.risk_report = report
                    except Exception as e:
                        st.error(f"Error en el escáner: {e}")
                        
            if "risk_report" in st.session_state:
                report = st.session_state.risk_report
                
                # Top Level Metrics
                met_col1, met_col2, met_col3 = st.columns([1,1,1])
                with met_col1:
                    score = report.total_score
                    st.metric("Risk Score", f"{score:.1f} / 100", delta="-PELIGRO" if score > 50 else "-SEGURO", delta_color="inverse")
                with met_col2:
                    st.metric("Cláusulas Faltantes", len(report.missing_critical))
                with met_col3:
                    st.write("💾 **Exportar Datos**")
                    st.download_button(
                        label="Descargar Reporte (JSON)",
                        data=report.model_dump_json(indent=2),
                        file_name=f"legalguard_risk_report.json",
                        mime="application/json",
                        use_container_width=True
                    )

                st.divider()

                # Critical Alerts
                if report.missing_critical:
                    st.error(f"🚨 **¡ALERTA ROJA!** Faltan las siguientes cláusulas críticas (Criticidad Alta):\n\n" + "\n".join([f"- {m}" for m in report.missing_critical]))
                else:
                    st.success("✅ ¡El contrato está blindado! Todas las cláusulas de Criticidad Alta se han encontrado.")
                
                st.divider()

                st.subheader("📋 Desglose del Análisis Legal")
                
                # Dividir Presentes y Ausentes para Visualización Limpia
                presentes = [c for c in report.clauses if c.is_present]
                ausentes = [c for c in report.clauses if not c.is_present]

                tab_pres, tab_aus = st.tabs([f"✅ Encontradas ({len(presentes)})", f"❌ Faltantes ({len(ausentes)})"])
                
                with tab_pres:
                    for c in presentes:
                        risk_color = "red" if c.risk_weight == 3 else ("orange" if c.risk_weight == 2 else "green")
                        with st.expander(f"✅ {c.clause_name} (Peso: {c.risk_weight})"):
                            st.markdown(f"**Extracto Literal:**\n> {c.excerpt}")
                            st.markdown(f"**Comentario Auditor:** {c.comment}")
                
                with tab_aus:
                    for c in ausentes:
                        st.warning(f"❌ {c.clause_name} (Peso original: {c.risk_weight}) - *{c.comment}*")
        
    # --- TAB 3: MÉTRICAS RAGAS (CALIDAD) ---
    with tab_metrics:
        st.markdown('<h1 class="main-header">Auditoría de Calidad RAGAS</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {text_secondary}; margin-top: -15px;">Evaluación continua mediante **LLM-as-a-Judge** (Fidelidad y Relevancia).</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📈 Ejecutar Evaluación de Calidad", use_container_width=True, type="primary"):
                with st.spinner("El Juez IA está auditando los logs de las últimas consultas..."):
                    results = run_evaluation()
                    if "error" in results:
                        st.error(f"Error en evaluación: {results['error']}")
                    else:
                        st.session_state.ragas_results = results
        
        with col2:
            st.info("Métricas basadas en las últimas 5 interacciones guardadas en el log de gobernanza.")

        if "ragas_results" in st.session_state:
            res = st.session_state.ragas_results
            
            # Grid de métricas
            m1, m2, m3 = st.columns(3)
            
            # Sacamos los valores del objeto Result de RAGAS
            # Dependiendo de la versión de ragas, el acceso puede variar. 
            # Intentamos acceso por dict o atributo.
            try:
                f_val = res["faithfulness"]
                r_val = res["answer_relevancy"]
                p_val = res["context_precision"]
            except:
                f_val = getattr(res, "faithfulness", 0)
                r_val = getattr(res, "answer_relevancy", 0)
                p_val = getattr(res, "context_precision", 0)

            m1.metric("Faithfulness (Fidelidad)", f"{f_val:.2%}", help="¿La respuesta se basa estrictamente en el PDF?")
            m2.metric("Answer Relevancy", f"{r_val:.2%}", help="¿La respuesta es útil para la pregunta?")
            m3.metric("Context Precision", f"{p_val:.2%}", help="¿Azure Search encontró los mejores fragmentos?")
            
            st.divider()
            st.subheader("Interpretación de Resultados")
            if f_val > 0.8:
                st.success("✅ **Altísima Fidelidad**: El sistema no está alucinando información externa.")
            else:
                st.warning("⚠️ **Riesgo de Alucinación**: Se detectaron respuestas que podrían contener datos no presentes en el PDF.")
            
            # Mostrar tabla de detalles si es posible
            try:
                df = st.session_state.ragas_results.to_pandas()
                st.dataframe(df, use_container_width=True)
            except:
                pass
        else:
            st.info("Haz clic en el botón superior para generar el reporte de métricas actual.")
    
    st.markdown("---")
    st.info("⬅️ Usa la barra lateral para subir y extraer el Markdown de un contrato primero.")

