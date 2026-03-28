import sys
import os
# Hackathon Fix: Añadimos la raíz del proyecto al sys.path para que Streamlit resuelva "src"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import httpx
import json
import asyncio
import tempfile
from typing import Generator
from datetime import datetime, timezone

import uuid
from azure.storage.blob import BlobServiceClient

# --- IMPORTACIONES DE LÓGICA DE NEGOCIO ---
from src.agent import LegalGuardAgent
from src.chat_history import (
    save_chat_session, 
    load_chat_session, 
    list_chat_sessions, 
    delete_chat_session
)
from src.ingestion.pipeline import (
    compute_file_hash, 
    check_duplicate_by_hash, 
    get_available_documents_enriched, 
    get_blob_sas_url
)
from src.metrics import (
    cargar_historial, 
    calcular_stats_historial, 
    cargar_ultima_evaluacion, 
    run_evaluation, 
    preparar_dataset_cuad, 
    eval_single_response
)
from src.comparator import compare_contract_versions

# --- CONFIGURACIÓN DE RUTAS GLOBALES ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
AUDIT_LOG_PATH = os.path.join(ROOT_DIR, "outputs/governance/audit_log.jsonl")

# --- MOTOR DE VISTA PREVIA & RIESGO (Global) ---
def get_preview_content_internal(selection):
    """Helper para obtener el contenido de vista previa según la selección."""
    if selection == "📄 Ingesta Actual":
        return st.session_state.get("md_content", None)
    else:
        try:
            from src.retrieval.search_engine import get_search_client
            s_client = get_search_client()
            results = list(s_client.search(
                search_text="*",
                filter=f"source_file eq '{selection}'",
                top=12,
                select=["content"]
            ))
            if results:
                return "\n\n".join([r["content"] for r in results])
        except Exception as e:
            return f"⚠️ Error recuperando '{selection}': {e}"
    return None

def render_legal_content_style(text, title="Documento"):
    """Formatea el contenido con estética Premium Legal."""
    import re
    # Importante: border_color y color_text deben ser globales o pasarse
    border_color = "#3B82F6" 
    color_text = "#FFFFFF" if st.session_state.get("dark_mode", False) else "#000000"
    
    content = text if text else "Sin contenido disponible."
    
    patterns = [
        r"(Article\s+\d+)", r"(Sección\s+\d+)", r"(SOP\s+No\.\s+\d+)", 
        r"(Cláusula\s+\d+)", r"(Standard\s+Operational\s+Procedure\s+No\.\s+\d+)"
    ]
    for p in patterns:
        content = re.sub(p, r'<span class="legal-section-badge">\1</span>', content, flags=re.IGNORECASE)

    return f"""
    <div class="legal-document-viewer">
        <div class="legal-document-card">
            <div style="margin-bottom: 20px; border-bottom: 1px solid {border_color}; padding-bottom: 10px;">
                <span style="font-weight: 700; color: #3B82F6; font-size: 1.1rem;">📄 {title}</span>
            </div>
            <div style="font-size: 0.95rem; color: {color_text};">
                {content}
            </div>
        </div>
    </div>
    """

# --- Utilidad de Sincronización Cloud (Opción B) ---
def upload_to_blob(file_bytes, file_name):
    """Sube el PDF original al contenedor de Azure Blob Storage."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "contratos-raw")
    
    if not conn_str:
        return False
        
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        blob_client.upload_blob(file_bytes, overwrite=True)
        return True
    except Exception as e:
        st.error(f"❌ Error en Sincronización Blob: {e}")
        return False

# 1. Configuración de página (SIEMPRE PRIMERO)
st.set_page_config(
    page_title="LegalGuard RAG | Smart Contract Review",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Estado de Sesión y Caching de Recursos
@st.cache_resource(ttl=3600)
def get_legal_agent(version_id: str = "1.3"):
    """Carga el agente con un ID de versión para forzar recargas si es necesario."""
    print(f"🚀 [DEBUG] Cargando Agente LegalGuard v{version_id}")
    return LegalGuardAgent()

@st.cache_resource
def get_search_client():
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    return SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "contratos-index"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "md_content" not in st.session_state:
    st.session_state.md_content = None

# Iniciar agente de forma perezosa (Solo cuando se necesite en el chat o scanner)
# st.session_state.agent = get_legal_agent()  <-- Eliminado de la carga global para velocidad

# Estado para documentos seleccionados en el Document Selector Pro
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []

# 3. Toggle de Modo Noche y Layout (Sidebar)
col_nav1, col_nav2 = st.sidebar.columns(2)
st.session_state.dark_mode = col_nav1.toggle("🌙 Noche", value=st.session_state.dark_mode)
if "pdf_collapsed" not in st.session_state:
    st.session_state.pdf_collapsed = False
st.session_state.pdf_collapsed = col_nav2.toggle("📄 Ocultar", value=st.session_state.pdf_collapsed)

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

# --- Inyección de CSS de Ultra-Velocidad (Hito Performance) ---
# Se inyecta aquí mismo, antes de cualquier importación pesada o lógica de negocio
# para que el cambio de Modo Noche sea casi instantáneo al ojo humano.
def get_custom_css(dark_mode, bg_app, bg_chat, color_text, color_header, border_color, sidebar_bg, input_bg, bubble_user, bubble_ai, text_secondary):
    return f"""
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

    /* Estilo Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
    }}

    /* Chat Bubbles - Glassmorphism Estilo Premium (Hito Frontend) */
    [data-testid="stChatMessage"] {{
        background: { "rgba(30, 41, 59, 0.7)" if dark_mode else "rgba(255, 255, 255, 0.7)" } !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid {border_color} !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-radius: 12px !important;
        margin-bottom: 12px;
        padding: 15px !important;
    }}

    .stChatMessage [data-testid="stMarkdownContainer"] p {{
        font-size: 0.95rem;
        line-height: 1.6;
    }}

    /* Estilo para las tarjetas de Documentos y Riesgos */
    .legal-document-card {{
        background: {bg_chat};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    /* Scrollbars Premium */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {border_color}; border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {color_header}; }}

    /* Botones y Inputs */
    .stButton > button {{
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }}
    
    .stTextInput input, .stTextArea textarea {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}

    /* Glassmorphism Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: { "rgba(30, 41, 59, 0.5)" if dark_mode else "rgba(255, 255, 255, 0.5)" };
        border: 1px solid {border_color};
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }}

    /* Badges Legales */
    .legal-section-badge {{
        background: {color_header};
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 4px;
    }}

    /* --- NUEVOS ESTILOS DASHBOARD PRO --- */
    .metric-card {{
        background: {bg_chat};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    .metric-card.faith::before  {{ background: #3B82F6; }}
    .metric-card.relev::before  {{ background: #10B981; }}
    .metric-card.prec::before   {{ background: #F59E0B; }}
    .metric-card.recall::before {{ background: #8B5CF6; }}

    .main-header {{
        font-size: 2.22rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, {color_header}, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.22rem !important;
    }}

    /* --- Citas Dinámicas con Resaltado al Pasar el Mouse --- */
    .cite-highlight {{
        background-color: rgba(59, 130, 246, 0.12);
        border-bottom: 2px dashed {color_header};
        cursor: pointer;
        position: relative;
        padding: 0 2px;
        transition: all 0.3s ease;
        text-decoration: none;
    }}
    .cite-highlight:hover {{
        background-color: rgba(59, 130, 246, 0.25);
    }}

    /* --- Tablas Premium para Signature Styles --- */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 0.88rem;
        background: {input_bg}44;
        border-radius: 8px;
        overflow: hidden;
    }}
    th {{
        background: {color_header}22;
        color: {color_header};
        font-weight: 700;
        text-align: left;
        padding: 10px;
        border-bottom: 2px solid {color_header}44;
    }}
    td {{
        padding: 10px;
        border-bottom: 1px solid {border_color};
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover {{ background: {color_header}11; }}
    .cite-highlight {{
        background-color: rgba(59, 130, 246, 0.12);
        border-bottom: 2px dashed {color_header};
        cursor: pointer;
        position: relative;
        padding: 0 2px;
        transition: all 0.3s ease;
        text-decoration: none;
    }}
    .cite-highlight:hover {{
        background-color: rgba(59, 130, 246, 0.25);
    }}
    
    /* Tooltip Premium con Retardo de 3 Segundos */
    .cite-highlight::after {{
        content: "Fragmento Original: \\A" attr(data-fragment);
        white-space: pre-wrap;
        position: absolute;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%);
        background: #111827;
        color: #F3F4F6;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 0.82rem;
        line-height: 1.4;
        width: 350px;
        visibility: hidden;
        opacity: 0;
        transition: opacity 0.4s ease 3s, visibility 0.4s ease 3s, transform 0.4s ease 3s; 
        z-index: 99999;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        border: 1px solid #374151;
        pointer-events: none;
    }}
    .cite-highlight:hover::after {{
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(-5px);
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
    }}

    #MainMenu, footer {{visibility: hidden;}}
</style>

<script>
    // Script de Sincronización Automática (Scroll Sync)
    document.addEventListener('click', function(e) {{
        const cite = e.target.closest('.cite-highlight');
        if (!cite) return;
        const fragment = cite.getAttribute('data-fragment');
        if (!fragment) return;
        const viewer = document.querySelector('.legal-document-card');
        if (!viewer) return;
        
        const searchText = fragment.substring(0, 60).toLowerCase();
        const walker = document.createTreeWalker(viewer, NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {{
            const node = walker.currentNode;
            if (node.textContent.toLowerCase().includes(searchText)) {{
                node.parentElement.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                node.parentElement.classList.add('sync-found');
                break;
            }}
        }}
    }});
</script>
"""

st.markdown(get_custom_css(
    st.session_state.dark_mode, bg_app, bg_chat, color_text, color_header, border_color, sidebar_bg, input_bg, bubble_user, bubble_ai, text_secondary
), unsafe_allow_html=True)

# 6. Backend Logic (SSE)
async def stream_from_backend(question: str, persona: str = "Orchestrator"):
    url = "http://localhost:8000/ask"
    payload = {
        "question": question, 
        "thread_id": "streamlit_user",
        "persona": persona
    }
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
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        
        # --- Validación de Duplicados (Opción B - SHA256) ✅ ---
        dup_check = check_duplicate_by_hash(file_hash)
        
        if dup_check["status"] == "duplicate":
            st.error(
                f"🔒 **Duplicado Detectado**: Este archivo ya existe en el índice como "
                f'`{dup_check["existing_file"]}`. No se realizará ninguna acción para evitar redundancia.'
            )
        else:
            if dup_check["status"] == "new_version":
                st.warning(
                    f"🔄 **Nueva Versión Detectada**: El nombre `{uploaded_file.name}` ya existe "
                    f"pero el contenido cambió. Se procederá a actualizar el índice."
                )

            with st.status("Preparando archivo local...", expanded=True) as status:
                try:
                    # 1. Guardar temporalmente
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    
                    # 2. Ingesta a Azure Document Intelligence
                    status.write("Enviando a Azure Document Intelligence...")
                    status.write("Extrayendo Markdown e inyectando Tablas HTML...")
                    
                    # Sincronización Automática con Blob Storage
                    if upload_to_blob(file_bytes, uploaded_file.name):
                        st.toast(f"☁️ {uploaded_file.name} sincronizado en el Storage", icon="✅")
                    
                    from src.ingestion.document_processor import extract_document_hybrid
                    from src.ingestion.pipeline import index_document_from_text
                    md_result = extract_document_hybrid(tmp_path)
                    
                    if md_result:
                        # 3. Indexación en Caliente con Hash adjunto 🔥
                        status.write("Enviando fragmentos a la Base Vectorial (Azure AI Search)...")
                        index_document_from_text(uploaded_file.name, md_result, file_hash=file_hash)
                        
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
    
    # --- Document Selector Pro ---
    st.subheader("🗂️ Documentos Activos")
    
    # Mostrar documentos seleccionados 
    if st.session_state.selected_docs:
        st.success(f"✅ {len(st.session_state.selected_docs)} doc(s) seleccionado(s)")
        for doc in st.session_state.selected_docs:
            st.caption(f"• {doc}")
    else:
        st.info("🌐 Búsqueda Global (todos los docs)")

    # --- Modal Emergente ---
    @st.dialog("🗂️ Gestionar Documentos", width="large")
    def show_document_selector():
        with st.spinner("Cargando documentos del índice..."):
            all_docs = get_available_documents_enriched()
        
        if not all_docs:
            st.warning("⚠️ No se encontraron documentos en el índice.")
            st.caption(f"Índice consultado: `{os.getenv('AZURE_SEARCH_INDEX_NAME', 'contratos-index')}`")
            st.info("⬆️ Sube un PDF desde el panel lateral para comenzar.")
            return
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent_docs = [d for d in all_docs if (d.get("upload_date") or "")[:10] == today]
        base_docs = all_docs  # Todos los documentos

        # INICIO: selección temporal dentro del modal
        temp_selected = list(st.session_state.selected_docs)

        tab_recent, tab_all = st.tabs([
            f"📅 Recientes ({len(recent_docs)})",
            f"🗄️ Base de Conocimiento ({len(base_docs)})"
        ])

        def render_doc_cards(docs, key_prefix):
            """Renderiza tarjetas de documentos con checkboxes, metadata y botón visor."""
            if not docs:
                st.info("💭 No hay documentos en esta categoría.")
                return
            
            for doc in docs:
                fname = doc["filename"]
                col_check, col_info, col_eye = st.columns([0.5, 8, 1])
                
                is_selected = fname in temp_selected
                checked = col_check.checkbox("Seleccionar", value=is_selected, key=f"{key_prefix}_{fname}", label_visibility="collapsed")
                
                if checked and fname not in temp_selected:
                    temp_selected.append(fname)
                elif not checked and fname in temp_selected:
                    temp_selected.remove(fname)
                
                with col_info:
                    st.markdown(f"**📄 {fname}**")
                    summary = doc.get("summary", "")
                    entities = doc.get("entities", "")
                    upload_date = (doc.get("upload_date") or "")[:10] or "Fecha no disponible"
                    
                    if summary:
                        st.caption(f"ℹ️ {summary}")
                    if entities:
                        # Mostrar entidades como tags visuales
                        entity_tags = " • ".join(entities.split(",")[:5])
                        st.markdown(f"<span style='font-size:0.75rem; color:#64748B;'>🏷️ {entity_tags}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:0.7rem; color:#94A3B8;'>📅 Subido: {upload_date}</span>", unsafe_allow_html=True)
                
                # Botón ojo: genera SAS URL y abre en nueva pestaña
                sas_url = get_blob_sas_url(fname)
                if sas_url:
                    col_eye.link_button("👁️", sas_url, help="Abrir PDF en nueva pestaña")
                
                st.divider()

        with tab_recent:
            render_doc_cards(recent_docs, "recent")
        
        with tab_all:
            render_doc_cards(base_docs, "all")
        
        # Botón de confirmación
        st.markdown("---")
        col_clear, col_confirm = st.columns([1, 2])
        
        if col_clear.button("🗑️ Limpiar Selección", use_container_width=True):
            st.session_state.selected_docs = []
            st.rerun()
        
        if col_confirm.button(
            f"✅ Usar {len(temp_selected)} documento(s) en la consulta RAG",
            type="primary",
            use_container_width=True,
            disabled=False
        ):
            st.session_state.selected_docs = temp_selected
            st.rerun()

    if st.button("🗂️ Gestionar Documentos", use_container_width=True):
        show_document_selector()

    selected_docs = st.session_state.selected_docs

    st.divider()
    
    # --- SWITCH DE AUDITORÍA HÍBRIDO ---
    st.subheader("🛡️ Modo de Vigilancia")
    realtime_audit = st.toggle(
        "Vigilancia en Tiempo Real",
        value=st.session_state.get("realtime_audit", False),
        help="ON: Cada respuesta se audita automáticamente con RAGAS. OFF: Botón manual '❓' debajo de cada respuesta."
    )
    st.session_state.realtime_audit = realtime_audit
    
    if realtime_audit:
        st.caption("🟢 Modo Activo: Cada respuesta será verificada por el Juez IA.")
    else:
        st.caption("⚪ Modo Manual: Usa el botón ❓ para auditar respuestas individuales.")

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
        st.session_state.pop("cosmos_session_id", None)
        st.rerun()

    st.divider()
    
    # --- PANEL DE HISTORIAL EN LA NUBE (Cosmos DB) ---
    st.subheader("☁️ Historial en la Nube")
    
    # Inicializar ID de sesión si no existe
    if "cosmos_session_id" not in st.session_state:
        st.session_state.cosmos_session_id = str(uuid.uuid4())[:8]
    
    st.caption(f"📌 Sesión actual: `{st.session_state.cosmos_session_id}`")
    
    col_save, col_new = st.columns(2)
    with col_save:
        if st.button("💾 Guardar", use_container_width=True):
            persona = st.session_state.get("selected_persona", "Orchestrator")
            success = save_chat_session(st.session_state.cosmos_session_id, st.session_state.messages, persona)
            if success:
                st.toast("✅ Sesión guardada en Azure Cosmos DB")
            else:
                st.toast("⚠️ No se pudo guardar (revisa COSMOS_CONNECTION_STRING)")
    with col_new:
        if st.button("🆕 Nueva", use_container_width=True):
            st.session_state.messages = []
            st.session_state.cosmos_session_id = str(uuid.uuid4())[:8]
            st.rerun()
    
    # Listar sesiones guardadas
    with st.expander("📂 Sesiones anteriores", expanded=False):
        sessions = list_chat_sessions(max_sessions=5)
        if sessions:
            for sess in sessions:
                s_id = sess.get("session_id", "?")
                s_msgs = sess.get("message_count", 0)
                s_persona = sess.get("persona", "")
                s_date = (sess.get("updated_at") or "")[:16]
                col_info, col_load, col_del = st.columns([4, 1.5, 1.5])
                col_info.markdown(f"**{s_id}** · {s_msgs} msgs · 🎭 {s_persona}\n\n_{s_date}_")
                if col_load.button("📂", key=f"load_{s_id}", help="Cargar sesión"):
                    loaded = load_chat_session(s_id)
                    if loaded:
                        st.session_state.messages = loaded.get("messages", [])
                        st.session_state.cosmos_session_id = s_id
                        st.toast(f"📂 Sesión {s_id} restaurada")
                        st.rerun()
                if col_del.button("🗑️", key=f"del_{s_id}", help="Eliminar sesión"):
                    delete_chat_session(s_id)
                    st.toast(f"🗑️ Sesión {s_id} eliminada")
                    st.rerun()
        else:
            st.info("Sin sesiones guardadas aún.")

# 8. Main Layout (Dinámico: 50/50 o 0/100)
if st.session_state.pdf_collapsed:
    col_pdf, col_chat = st.columns([0.01, 10]) # Casi oculto pero existente para evitar bugs de flujo
else:
    col_pdf, col_chat = st.columns([5, 5])

with col_pdf:
    # Inicializar variables para evitar NameError cuando está colapsado o no hay docs
    compare_mode = False
    base_doc_selection = None
    ref_doc_selection = None
    
    if not st.session_state.pdf_collapsed:
        st.markdown(f'<h3 style="color: {color_text}; font-weight: 700; display: flex; align-items: center; gap: 8px;">📄 Mesa de Trabajo Inteligente</h3>', unsafe_allow_html=True)
        
        # --- COMPARADOR DE VERSIONES (sección superior) ---
        with st.expander("🔀 Comparador de Versiones de Contratos", expanded=False):
            all_doc_names = [d["filename"] for d in get_available_documents_enriched()]
            if len(all_doc_names) < 2:
                st.info("Necesitas al menos 2 documentos indexados para comparar versiones.")
            else:
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    doc_v1 = st.selectbox("📄 Versión Anterior", all_doc_names, key="compare_v1")
                with col_v2:
                    other_docs = [d for d in all_doc_names if d != doc_v1]
                    doc_v2 = st.selectbox("📄 Versión Nueva", other_docs, key="compare_v2") if other_docs else None
                
                if doc_v2 and st.button("🔍 Comparar Contratos", use_container_width=True, type="primary"):
                    with st.spinner(f"Analizando diferencias entre {doc_v1} y {doc_v2}..."):
                        compare_result = compare_contract_versions(doc_v1, doc_v2)
                        if "error" in compare_result:
                            st.error(compare_result["error"])
                        else:
                            st.success(f"✅ {compare_result.get('resumen', 'Comparación completada')}")
                            # ... (render logic for changes)
        
        # Fuentes disponibles: El subido actualmente + los seleccionados en el filtro
        available_for_preview = []
        if st.session_state.md_content:
            available_for_preview.append("📄 Ingesta Actual")
        if selected_docs:
            available_for_preview.extend(selected_docs)
        
        if available_for_preview:
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                base_doc_selection = st.selectbox("Documento Principal", available_for_preview, index=0)
            with col_ctrl2:
                compare_mode = st.toggle("⚖️ Comparar", value=False)
            
            ref_doc_selection = None
            if compare_mode:
                ref_doc_selection = st.selectbox(
                    "Documento de Referencia",
                    [d for d in available_for_preview if d != base_doc_selection],
                    index=0 if len(available_for_preview) > 1 else None
                )
            
            # Persistir selección para el Risk Scanner
            st.session_state.last_previewed_doc = base_doc_selection

        # Renderizado de la Mesa de Trabajo Premium
        bg_info = "#1E293B" if st.session_state.dark_mode else "#E0F2FE"
        color_info = "#3B82F6" if st.session_state.dark_mode else "#0369A1"
        
        st.markdown(f"""
            <div style="background-color: {bg_info}; border-left: 4px solid #3B82F6; padding: 10px; border-radius: 8px; font-size: 0.85rem; color: {color_info}; margin-bottom: 15px;">
                💡 <b>Mesa de Trabajo:</b> {'Comparando dos documentos en paralelo' if compare_mode else 'Analizando documento individual'}.
            </div>
        """, unsafe_allow_html=True)

        if compare_mode and ref_doc_selection:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(render_legal_content_style(get_preview_content_internal(base_doc_selection), base_doc_selection), unsafe_allow_html=True)
            with col_right:
                st.markdown(render_legal_content_style(get_preview_content_internal(ref_doc_selection), ref_doc_selection), unsafe_allow_html=True)
        else:
            st.markdown(render_legal_content_style(get_preview_content_internal(base_doc_selection), base_doc_selection), unsafe_allow_html=True)

    elif not st.session_state.pdf_collapsed:
        st.markdown(f"""
            <div class="pdf-viewer-placeholder">
                <div style="text-align: center;">
                    <h4 style="margin: 0; color: {color_text};">Mesa de Trabajo Vacía</h4>
                    <p style="font-size: 0.9rem; margin-top: 5px; color: {text_secondary};">Sube un PDF o selecciona documentos en el panel lateral para activar la vista previa.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

with col_chat:
    tab_chat, tab_risk, tab_metrics, tab_command = st.tabs(["💬 Asistente Legal", "📋 Risk Scanner (CUAD)", "📊 Métricas RAGAS", "📡 Centro de Comando"])
    
    with tab_chat:
        st.markdown('<h1 class="main-header">Agente LegalGuard</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {text_secondary}; margin-top: -15px;">Interacción conversacional (LangGraph RAG Activo).</p>', unsafe_allow_html=True)
        
        # --- Selector de Persona (Rol del Profesional) ---
        persona_col, _ = st.columns([2, 3])
        with persona_col:
            active_persona = st.selectbox(
                "🎭 Mi perfil profesional",
                ["Legal", "Financiero", "Salud", "Ejecutivo", "Orchestrator"],
                index=0,
                key="selected_persona",
                help="Las respuestas del LLM se adaptarán a tu rol."
            )
        
        # --- RENDERIZADO DE CHAT (Limpio y Único) ---
        mensajes = st.session_state.messages

        def render_assistant_content(msg, idx):
            """Helper para renderizar la respuesta del asistente con su estilo."""
            p = msg.get("persona", "Orchestrator")
            icon = {"Salud": "🏥", "Legal": "🛡️", "Financiero": "💰", "Ejecutivo": "📌"}.get(p, "🎭")
            badge_color = {"Salud": "#10B981", "Legal": "#3B82F6", "Financiero": "#F59E0B", "Ejecutivo": "#8B5CF6"}.get(p, "#64748B")
            st.markdown(f'<span style="font-size: 0.7rem; background: {badge_color}22; color: {badge_color}; padding: 2px 10px; border: 1px solid {badge_color}; border-radius: 12px; font-weight: 600;">{icon} {p}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"], unsafe_allow_html=True)
            
            # Auditoría y Docs Colapsables
            if "documents" in msg and msg["documents"]:
                with st.expander("🔍 Ver fragmentos originales y fuentes", expanded=False):
                    for j, doc in enumerate(msg["documents"]):
                        st.markdown(f"**Fragmento {j+1} - {doc['source_file']}**")
                        st.info(doc["content"])
            else:
                st.caption("ℹ️ *Auditoría no disponible: Cálculo Interno.*")

        # 1. Historial en Expander (Todo menos el último bloque)
        if len(mensajes) > 2:
            with st.expander(f"📖 Ver historial anterior ({len(mensajes)//2 - 1} interacciones)", expanded=False):
                for i in range(len(mensajes) - 2):
                    msg = mensajes[i]
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "assistant":
                            st.markdown(msg["content"], unsafe_allow_html=True)
                        else:
                            st.markdown(msg["content"])

        # 2. Último Bloque (Siempre visible)
        actual = mensajes[-2:] if len(mensajes) >= 2 else mensajes
        for i, msg in enumerate(actual):
            real_idx = len(mensajes) - len(actual) + i
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    render_assistant_content(msg, real_idx)
                else:
                    st.markdown(msg["content"])

        # --- PROCESAMIENTO DE INPUT ---
        if st.session_state.md_content or selected_docs:
            if prompt := st.chat_input("Escribe tu consulta aquí..."):
                # 1. Guardar mensaje del usuario
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # 2. Mostrar estado mientras se procesa
                with st.status("LegalGuard está analizando...", expanded=True) as status:
                    if "agent" not in st.session_state:
                         st.session_state.agent = get_legal_agent(version_id="1.4")
                    
                    persona_actual = st.session_state.get("selected_persona", "Orchestrator")
                    result = st.session_state.agent.run(prompt, filter_docs=selected_docs, persona=persona_actual)
                    
                    # 3. Guardar respuesta del asistente
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "documents": result["documents"],
                        "persona": persona_actual
                    })
                    status.update(label="Análisis Finalizado", state="complete")
                
                # 4. RERUN para renderizar limpiamente
                st.rerun()
        else:
            st.info("⬅️ Sube un PDF o selecciona documentos en el **Filtro de Memoria** para comenzar.")

    with tab_risk:
        st.markdown('<h1 class="main-header">Scanner de Riesgo Global</h1>', unsafe_allow_html=True)
        
        # Determinar qué contenido auditar (Ingesta actual o el seleccionado en la mesa)
        target_doc = st.session_state.get("last_previewed_doc", "📄 Ingesta Actual")
        content_to_scan = get_preview_content_internal(target_doc)
        
        if content_to_scan:
            st.info(f"📋 Documento listo para auditoría: **{target_doc}**")
            if st.button("🔎 Ejecutar Auditoría Profunda (CUAD)", use_container_width=True, type="primary"):
                with st.status(f"Analizando riesgos en {target_doc}...", expanded=True):
                    try:
                        from src.risk_scanner import scan_contract
                        st.session_state.risk_report = scan_contract(content_to_scan)
                    except Exception as e: 
                        st.error(f"Error en el escáner: {e}")
        else:
            st.warning("⚠️ No hay contenido cargado para auditar. Selecciona un documento en la Mesa de Trabajo.")

        if "risk_report" in st.session_state:
                report = st.session_state.risk_report
                met_col1, met_col2, met_col3 = st.columns([1,1,1])
                with met_col1: st.metric("Risk Score", f"{report.total_score:.1f} / 100")
                with met_col2: st.metric("Cláusulas Faltantes", len(report.missing_critical))
                with met_col3: st.download_button("Descargar Reporte (JSON)", data=report.model_dump_json(indent=2), file_name="legalguard_risk_report.json", mime="application/json", use_container_width=True)
                st.divider()
                if report.missing_critical: st.error(f"🚨 **¡ALERTA ROJA!** Faltan cláusulas críticas:\n\n" + "\n".join([f"- {m}" for m in report.missing_critical]))
                else: st.success("✅ ¡El contrato está blindado!")
                st.divider()
                st.subheader("📋 Desglose del Análisis Legal")
                presentes = [c for c in report.clauses if c.is_present]
                ausentes = [c for c in report.clauses if not c.is_present]
                tab_pres, tab_aus = st.tabs([f"✅ Encontradas ({len(presentes)})", f"❌ Faltantes ({len(ausentes)})"])
                with tab_pres:
                    for c in presentes:
                        with st.expander(f"✅ {c.clause_name}"):
                            st.markdown(f"**Extracto:**\n> {c.excerpt}\n**Comentario:** {c.comment}")
                with tab_aus:
                    for c in ausentes: 
                        st.warning(f"❌ {c.clause_name} - *{c.comment}*")
                    if not ausentes:
                        st.info("✅ No se detectaron cláusulas faltantes.")
        else:
            st.info("📊 Sube un contrato y haz clic en 'Ejecutar Auditoría' para ver el análisis de riesgo.")
        
    with tab_metrics:
        st.markdown('<h1 class="main-header">Centro de Calidad & Trazabilidad</h1>', unsafe_allow_html=True)
        eval_data = cargar_ultima_evaluacion()
        scores = eval_data.get("scores", {})
        is_benchmark = eval_data.get("is_benchmark", False)
        
        m1, m2, m3, m4 = st.columns(4)
        def render_metric_card(col, label, val, desc, mode_active=True):
            if not mode_active:
                val_str, cbar = "N/A", "#94A3B8"
            else:
                val_str = f"{val:.0%}"
                cbar = "#10B981" if val >= 0.8 else ("#F59E0B" if val >= 0.6 else "#EF4444")
            with col:
                st.markdown(f'<div class="metric-card"><div style="font-size: 0.75rem; color: #64748B;">{label}</div><div style="font-size: 1.5rem; font-weight: 700; color: {cbar};">{val_str}</div><div style="font-size: 0.65rem; color: #94A3B8;">{desc}</div></div>', unsafe_allow_html=True)
        
        render_metric_card(m1, "Fidelidad", scores.get("faithfulness", 0), "Bases documentales.")
        render_metric_card(m2, "Relevancia", scores.get("answer_relevancy", 0), "Respuesta directa.")
        render_metric_card(m3, "Precisión", scores.get("context_precision", 0), "Calidad Azure Search.", mode_active=is_benchmark)
        render_metric_card(m4, "Recall", scores.get("context_recall", 0), "Cobertura de info.", mode_active=is_benchmark)
        
        if not is_benchmark:
            st.info("💡 **Nota**: Precisión y Recall solo se activan en modo **Benchmark** (con Ground Truth). En **Auditoría en Vivo**, evaluamos la fidelidad y relevancia de tus interacciones reales.")
        
        st.divider()
        st.subheader("⚖️ Auditoría en Vivo (RAGAS)")
        st.caption("Evalúa la calidad de las últimas interacciones reales en el chat.")
        if st.button("⚖️ Auditar Historial de Chat", use_container_width=True, type="primary"):
            with st.status("El Juez IA está auditando el historial...", expanded=True):
                run_evaluation(max_samples=5)
            st.rerun()

    with tab_command:
        st.markdown('<h1 class="main-header">Centro de Comando</h1>', unsafe_allow_html=True)
        st.subheader("🫀 Estado de Servicios")
        if st.button("🔄 Check Health", use_container_width=True):
            from src.telemetry import check_azure_health
            st.session_state.azure_health = check_azure_health()
        if "azure_health" in st.session_state:
            health = st.session_state.azure_health
            cols = st.columns(len(health))
            for i, (service, data) in enumerate(health.items()):
                with cols[i]: st.markdown(f"**{service}**\n{data['icon']} {data['status']}")
        st.divider()
        st.divider()
        st.subheader("📊 Consumo de Tokens (Acumulado)")
        historial = cargar_historial()
        if historial:
            df_tokens = pd.DataFrame([
                {"Fecha": r["timestamp"][:16].replace("T", " "), "Tokens": r.get("tokens", {}).get("total_tokens", 0)}
                for r in historial if "tokens" in r
            ])
            if not df_tokens.empty:
                st.bar_chart(df_tokens.set_index("Fecha"))
                st.caption(f"Total tokens consumidos en esta sesión: {df_tokens['Tokens'].sum():,}")
            else:
                st.info("No hay datos de consumo registrados todavía.")
        else:
            st.info("Inicia una conversación para ver el consumo de tokens.")

    st.markdown("---")
    with st.expander("🛡️ Trazabilidad y Compliance"):
        if os.path.exists(AUDIT_LOG_PATH):
            with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
                audit_data = f.read()
            st.download_button(
                label="📂 Descargar Log Completo (JSONL)",
                data=audit_data,
                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.jsonl",
                mime="application/jsonl",
                use_container_width=True
            )
        else:
            st.error(f"Archivo de auditoría no encontrado en: {AUDIT_LOG_PATH}")
            if st.button("🔧 Re-inicializar Auditoría"):
                from src.governance import GovernanceManager
                GovernanceManager(log_path=AUDIT_LOG_PATH)
                st.rerun()

    if not (st.session_state.md_content or selected_docs):
        st.info("⬅️ Sube un PDF o selecciona documentos en el **Filtro de Memoria** para comenzar.")
    else:
        st.success("✅ Sistema listo para consultas legales.")

