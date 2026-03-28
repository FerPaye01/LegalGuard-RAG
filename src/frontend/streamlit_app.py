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
from datetime import datetime, timezone

# Import the Document Intelligence Script
from azure.storage.blob import BlobServiceClient
from src.ingestion.document_processor import extract_document_hybrid
from src.ingestion.pipeline import (
    index_document_from_text, compute_file_hash, check_duplicate_by_hash,
    get_available_documents_enriched, get_blob_sas_url
)
from src.agent import LegalGuardAgent
from src.risk_scanner import scan_contract
from src.metrics import run_evaluation, eval_single_response
from src.comparator import compare_contract_versions

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
@st.cache_resource
def get_legal_agent():
    """Inicializa el cerebro de LangGraph una sola vez y lo mantiene en memoria."""
    return LegalGuardAgent()

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
# 5. Optimización de Estilos
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

    /* Chat Bubbles */
    [data-testid="stChatMessage"] {{
        background-color: {bg_chat} !important;
        border: 1px solid {border_color};
        border-radius: 12px;
        margin-bottom: 15px;
        padding: 15px;
    }}

    /* Header e Inputs */
    .main-header {{
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #2DD4BF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }}

    /* Tables support for Doc Intel */
    table {{
        width: 100%;
        border-collapse: collapse;
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

    /* PREMIUM LEGAL VIEW INTERFACE */
    .legal-document-viewer {{
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 15px;
        scrollbar-width: thin;
        scrollbar-color: #3B82F6 {bg_app};
    }}
    
    .legal-document-viewer::-webkit-scrollbar {{
        width: 6px;
    }}
    .legal-document-viewer::-webkit-scrollbar-thumb {{
        background: #3B82F6;
        border-radius: 10px;
    }}

    .legal-document-card {{
        background-color: {bg_chat};
        border: 1px solid {border_color};
        border-left: 5px solid #3B82F6;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        line-height: 1.7;
        color: {color_text};
    }}

    .legal-section-badge {{
        display: inline-block;
        padding: 2px 10px;
        background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
        color: white !important;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }}

    .legal-text-highlight {{
        background-color: rgba(59, 130, 246, 0.1);
        border-bottom: 2px solid #3B82F6;
        font-weight: 600;
        padding: 0 2px;
    }}

    /* SISTEMA DE CITAS CON TOOLTIP + SCROLL SYNC (Efecto "Wow") */
    .cite-highlight {{
        background-color: rgba(255, 215, 0, 0.25);
        border-bottom: 2px solid #FFD700;
        cursor: pointer;
        position: relative;
        padding: 0 4px;
        border-radius: 3px;
        transition: background-color 0.3s ease;
    }}

    .cite-highlight:hover {{
        background-color: rgba(255, 215, 0, 0.5);
    }}

    /* Tooltip (aparece tras 3 segundos de hover) */
    .cite-highlight::after {{
        content: '🔗 Click para localizar en documento. Mantén hover para ver la cita original: ' attr(data-fragment);
        position: absolute;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #0F172A;
        color: #E2E8F0;
        padding: 12px 16px;
        border-radius: 10px;
        width: max-content;
        max-width: 350px;
        font-size: 0.82em;
        line-height: 1.5;
        border: 1px solid #3B82F6;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        white-space: pre-wrap;
        font-style: italic;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease 2s, visibility 0.3s ease 2s;
        z-index: 9999;
        pointer-events: none;
    }}

    .cite-highlight:hover::after {{
        opacity: 1;
        visibility: visible;
    }}

    .cite-highlight::before {{
        content: '';
        position: absolute;
        bottom: 118%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #3B82F6;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease 2s, visibility 0.3s ease 2s;
        z-index: 9999;
    }}

    .cite-highlight:hover::before {{
        opacity: 1;
        visibility: visible;
    }}

    /* Animación de parpadeo para el fragmento encontrado en la Mesa de Trabajo */
    @keyframes sync-blink {{
        0%, 100% {{ background-color: transparent; }}
        25% {{ background-color: rgba(59, 130, 246, 0.4); }}
        50% {{ background-color: transparent; }}
        75% {{ background-color: rgba(59, 130, 246, 0.4); }}
    }}

    .sync-found {{
        animation: sync-blink 1.5s ease 2;
        border-left: 4px solid #3B82F6 !important;
        padding-left: 8px !important;
        border-radius: 4px;
        scroll-margin-top: 80px;
    }}
</style>

<!-- Script de Sincronización Automática (Scroll Sync) -->
<script>
document.addEventListener('click', function(e) {{
    const cite = e.target.closest('.cite-highlight');
    if (!cite) return;
    
    const fragment = cite.getAttribute('data-fragment');
    if (!fragment) return;
    
    // Buscar en la Mesa de Trabajo (legal-document-card)
    const viewer = document.querySelector('.legal-document-card');
    if (!viewer) return;
    
    // Limpiar highlights anteriores
    document.querySelectorAll('.sync-found').forEach(el => {{
        el.classList.remove('sync-found');
    }});
    
    // Buscar el texto en el visor de documentos
    const searchText = fragment.substring(0, 60).toLowerCase();
    const walker = document.createTreeWalker(viewer, NodeFilter.SHOW_TEXT);
    let found = false;
    
    while (walker.nextNode()) {{
        const node = walker.currentNode;
        if (node.textContent.toLowerCase().includes(searchText)) {{
            // Envolver el nodo padre con la clase de sincronización
            const parent = node.parentElement;
            parent.classList.add('sync-found');
            
            // Scroll suave hacia el fragmento
            parent.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            
            // Quitar la clase después de 4 segundos
            setTimeout(() => {{
                parent.classList.remove('sync-found');
            }}, 4000);
            
            found = true;
            break;
        }}
    }}
    
    if (!found) {{
        // Si no encontró el texto exacto, hacer scroll al inicio del visor
        viewer.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
}});
</script>
"""

st.markdown(get_custom_css(
    st.session_state.dark_mode, bg_app, bg_chat, color_text, color_header, border_color, sidebar_bg, input_bg, bubble_user, bubble_ai, text_secondary
), unsafe_allow_html=True)

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
        recent_docs = [d for d in all_docs if d.get("upload_date", "")[:10] == today]
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
                checked = col_check.checkbox("", value=is_selected, key=f"{key_prefix}_{fname}")
                
                if checked and fname not in temp_selected:
                    temp_selected.append(fname)
                elif not checked and fname in temp_selected:
                    temp_selected.remove(fname)
                
                with col_info:
                    st.markdown(f"**📄 {fname}**")
                    summary = doc.get("summary", "")
                    entities = doc.get("entities", "")
                    upload_date = doc.get("upload_date", "")[:10] or "Fecha no disponible"
                    
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
        st.rerun()

# 8. Main Layout (Opción B: 50/50)
col_pdf, col_chat = st.columns([5, 5])

with col_pdf:
    st.markdown(f'<h3 style="color: {color_text}; font-weight: 700; display: flex; align-items: center; gap: 8px;">📄 Mesa de Trabajo Inteligente</h3>', unsafe_allow_html=True)
    
    # --- COMPARADOR DE VERSIONES (sección superior) ---
    with st.expander("🔀 Comparador de Versiones de Contratos", expanded=False):
        all_doc_names = [d["name"] for d in get_available_documents_enriched()]
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
                        cambios = compare_result.get("cambios", [])
                        if cambios:
                            for cambio in cambios:
                                tipo = cambio.get("tipo", "modificado")
                                impacto = cambio.get("impacto", "medio")
                                color = {"nuevo": "#10B981", "eliminado": "#EF4444", "modificado": "#F59E0B"}.get(tipo, "#64748B")
                                icono = {"nuevo": "➕", "eliminado": "➖", "modificado": "⚡"}.get(tipo, "•")
                                impact_badge = {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}.get(impacto, "🟡")
                                st.markdown(f"""
                                    <div style="border-left: 4px solid {color}; padding: 10px 15px; margin: 8px 0; background: rgba(0,0,0,0.1); border-radius: 6px;">
                                        <b>{icono} {cambio.get('clausula', 'Cláusula')} {impact_badge} Impacto {impacto.upper()}</b><br>
                                        <i style="color:#94A3B8; font-size:0.85rem;">Antes: {str(cambio.get('antes', 'N/A'))[:150]}...</i><br>
                                        <span style="font-size:0.85rem;">Después: {str(cambio.get('despues', 'N/A'))[:150]}...</span>
                                    </div>
                                """, unsafe_allow_html=True)
    
    # Fuentes disponibles: El subido actualmente + los seleccionados en el filtro
    available_for_preview = []
    if st.session_state.md_content:
        available_for_preview.append("📄 Ingesta Actual")
    if selected_docs:
        available_for_preview.extend(selected_docs)
    
    if available_for_preview:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        
        with col_ctrl1:
            base_doc_selection = st.selectbox(
                "Documento Principal", 
                available_for_preview, 
                index=0,
                help="Selecciona el documento que deseas analizar principalmente."
            )
        
        with col_ctrl2:
            compare_mode = st.toggle("⚖️ Comparar", value=False, help="Activa la vista dividida para contrastar dos contratos.")
        
        ref_doc_selection = None
        if compare_mode:
            ref_doc_selection = st.selectbox(
                "Documento de Referencia",
                [d for d in available_for_preview if d != base_doc_selection],
                index=0 if len(available_for_preview) > 1 else None,
                help="Selecciona el documento contra el cual quieres comparar."
            )

        def render_legal_content(text, title="Documento"):
            """Formatea el contenido con estética Premium Legal."""
            import re
            
            # Limpieza y preparación
            content = text if text else "Sin contenido disponible."
            
            # Resaltado de Secciones/Artículos (Regex simple)
            # Busca patrones como "Article 1", "Sección 2", "SOP No. 5", "Cláusula 4"
            patterns = [
                r"(Article\s+\d+)", r"(Sección\s+\d+)", r"(SOP\s+No\.\s+\d+)", 
                r"(Cláusula\s+\d+)", r"(Standard\s+Operational\s+Procedure\s+No\.\s+\d+)"
            ]
            for p in patterns:
                content = re.sub(p, r'<span class="legal-section-badge">\1</span>', content, flags=re.IGNORECASE)

            html = f"""
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
            return html

        def get_preview_content(selection):
            """Helper para obtener el contenido de vista previa según la selección."""
            if selection == "📄 Ingesta Actual":
                return st.session_state.md_content
            else:
                try:
                    from azure.search.documents import SearchClient
                    from azure.core.credentials import AzureKeyCredential
                    s_client = SearchClient(
                        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
                        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "contratos-index"),
                        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
                    )
                    results = list(s_client.search(
                        search_text="*",
                        filter=f"source_file eq '{selection}'",
                        top=12, # Aumentamos el contexto para el nuevo visor
                        select=["content"]
                    ))
                    if results:
                        return "\n\n".join([r["content"] for r in results])
                except Exception as e:
                    return f"⚠️ Error recuperando '{selection}': {e}"
            return None

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
                content_base = get_preview_content(base_doc_selection)
                st.markdown(render_legal_content(content_base, base_doc_selection), unsafe_allow_html=True)
            with col_right:
                content_ref = get_preview_content(ref_doc_selection)
                st.markdown(render_legal_content(content_ref, ref_doc_selection), unsafe_allow_html=True)
        else:
            content_base = get_preview_content(base_doc_selection)
            st.markdown(render_legal_content(content_base, base_doc_selection), unsafe_allow_html=True)

    else:
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
                ["Legal", "Financiero", "Salud", "Orchestrator"],
                index=0,
                help="Las respuestas del LLM se adaptarán a tu rol."
            )
        
        # --- Historial de Chat con Acordeón Comprimido ---
        mensajes = st.session_state.messages
        
        # Función helper para mostrar badges de auditoría
        def render_audit_badge(msg, msg_idx):
            """Muestra el badge de auditoría o botón manual según el modo."""
            score = msg.get("audit_score")
            
            if score and score.get("status") == "ok":
                f_val = score.get("faithfulness", 0)
                r_val = score.get("answer_relevancy", 0)
                badge = "✅" if f_val > 0.8 else ("⚠️" if f_val > 0.5 else "🚨")
                st.caption(f"{badge} Fidelidad: {f_val:.0%} | Relevancia: {r_val:.0%}")
            elif not st.session_state.get("realtime_audit", False) and msg.get("documents"):
                if st.button("❓ Auditar esta respuesta", key=f"audit_btn_{msg_idx}"):
                    with st.spinner("🛡️ Juez IA verificando..."):
                        contexts = [d["content"] for d in msg["documents"] if d.get("content")]
                        # Buscar la pregunta anterior
                        q_idx = msg_idx - 1
                        question = mensajes[q_idx]["content"] if q_idx >= 0 else "N/A"
                        result_audit = eval_single_response(question, msg["content"], contexts)
                        msg["audit_score"] = result_audit
                        st.rerun()
        
        if len(mensajes) > 2:
            historial = mensajes[:-2]
            ultima_interaccion = mensajes[-2:]
            
            with st.expander(f"📖 Historial ({len(historial)//2 if len(historial) >= 2 else len(historial)} interaccion/es anteriores)", expanded=False):
                for i in range(0, len(historial) - 1, 2):
                    user_msg = historial[i]["content"]
                    ai_msg = historial[i+1]
                    with st.expander(f"🗣️ {user_msg[:60]}{'...' if len(user_msg) > 60 else ''}"):
                        st.markdown(ai_msg["content"], unsafe_allow_html=True)
                        render_audit_badge(ai_msg, i+1)
            
            # Mostrar última interacción completa
            with st.chat_message(ultima_interaccion[0]["role"]):
                st.markdown(ultima_interaccion[0]["content"])
            if len(ultima_interaccion) > 1:
                last_msg = ultima_interaccion[1]
                last_idx = len(mensajes) - 1
                with st.chat_message(last_msg["role"]):
                    st.markdown(last_msg["content"], unsafe_allow_html=True)
                    render_audit_badge(last_msg, last_idx)
                    if "documents" in last_msg:
                        with st.expander("🔍 Ver fragmentos originales y fuentes"):
                            for i, doc in enumerate(last_msg["documents"]):
                                st.markdown(f"**Fragmento {i+1} - {doc['source_file']}**")
                                st.info(doc["content"])
        else:
            # Menos de 2 mensajes: mostrar normalmente
            for idx, msg in enumerate(mensajes):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg["role"] == "assistant":
                        render_audit_badge(msg, idx)
                        if "documents" in msg:
                            with st.expander("🔍 Ver fragmentos originales y fuentes"):
                                for i, doc in enumerate(msg["documents"]):
                                    st.markdown(f"**Fragmento {i+1} - {doc['source_file']}**")
                                    st.info(doc["content"])

        # Habilitar chat si hay contenido subido O si hay docs seleccionados en el filtro
        if st.session_state.md_content or selected_docs:
            if prompt := st.chat_input("Pregunta sobre las cláusulas u objetos del contrato..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.status("Analizando requerimiento legal...", expanded=True) as status:
                        # CARGA PEREZOSA: Inicializar agente solo ahora
                        if "agent" not in st.session_state:
                            st.session_state.agent = get_legal_agent()
                        
                        st.write("🔍 Clasificando intención con el **Router**...")
                        
                        # Llamada al agente con persona
                        persona_actual = st.session_state.get("selected_persona", "Orchestrator")
                        result = st.session_state.agent.run(prompt, filter_docs=selected_docs, persona=persona_actual)
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
                    
                    st.markdown(final_text, unsafe_allow_html=True)
                    
                    # --- AUDITORÍA HÍBRIDA: Modo Real-Time ---
                    audit_score = None
                    if st.session_state.get("realtime_audit", False) and docs:
                        with st.spinner("🛡️ Juez IA verificando fidelidad..."):
                            contexts = [d["content"] for d in docs if d.get("content")]
                            audit_score = eval_single_response(prompt, final_text, contexts)
                        
                        f_val = audit_score.get("faithfulness", 0)
                        r_val = audit_score.get("answer_relevancy", 0)
                        
                        if f_val > 0.8:
                            st.success(f"✅ Verificado: Faithfulness {f_val:.0%} | Relevancy {r_val:.0%}")
                        elif f_val > 0.5:
                            st.warning(f"⚠️ Precaución: Faithfulness {f_val:.0%} | Relevancy {r_val:.0%}")
                        else:
                            st.error(f"🚨 Riesgo: Faithfulness {f_val:.0%} | Relevancy {r_val:.0%}")
                    
                    # --- TELEMETRÍA INLINE (Mini-barra de latencia por nodo) ---
                    telemetry = result.get("telemetry", {})
                    nodes = telemetry.get("nodes", {})
                    if nodes:
                        total_ms = telemetry.get("total_ms", 1)
                        with st.expander(f"⏱️ Telemetría del Grafo ({total_ms}ms total)", expanded=False):
                            for node_name, ms in nodes.items():
                                pct = min(int((ms / total_ms) * 100), 100)
                                icon = {
                                    "router": "🧠", "retriever": "🔍", 
                                    "grader": "⚖️", "generator": "✍️"
                                }.get(node_name, "⚙️")
                                st.markdown(f"{icon} **{node_name.capitalize()}**: {ms}ms")
                                st.progress(pct / 100)
                    
                    if docs:
                        with st.expander("🔍 Ver fragmentos originales y fuentes"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Fragmento {i+1} - {doc['source_file']}**")
                                st.info(doc["content"])

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_text,
                        "documents": docs,
                        "telemetry": telemetry,
                        "audit_score": audit_score
                    })
        else:
            st.info("⬅️ Sube un PDF o selecciona documentos en el **Filtro de Memoria** para comenzar.")

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
            max_samples = st.slider("Cantidad de mensajes a evaluar", 1, 20, 5, help="Determina cuántas interacciones recientes del log serán auditadas por el Juez IA.")
            if st.button("📈 Ejecutar Evaluación de Calidad", use_container_width=True, type="primary"):
                with st.spinner(f"El Juez IA está auditando los últimos {max_samples} logs..."):
                    results = run_evaluation(max_samples=max_samples)
                    if "error" in results:
                        st.error(f"Error en evaluación: {results['error']}")
                    else:
                        st.session_state.ragas_results = results
        
        with col2:
            st.info(f"Métricas basadas en las últimas {max_samples} interacciones guardadas en el log de gobernanza.")

        @st.dialog("🔍 Inspección de Fidelidad (Muestra Individual)", width="large")
        def show_sample_details(sample):
            st.markdown(f"### ❓ Pregunta del Usuario")
            st.info(sample.get("question", "N/A"))
            
            st.markdown(f"### 📄 Contexto Recuperado (Azure Search)")
            for i, ctx in enumerate(sample.get("contexts", [])):
                with st.expander(f"Fragmento {i+1}"):
                    st.markdown(ctx)
            
            st.markdown(f"### 🤖 Respuesta de la IA")
            st.success(sample.get("answer", "N/A"))
            
            st.divider()
            st.markdown("### 📊 Métricas de esta Muestra")
            m1, m2 = st.columns(2)
            m1.metric("Faithfulness", f"{sample.get('faithfulness', 0):.2%}")
            m2.metric("Answer Relevancy", f"{sample.get('answer_relevancy', 0):.2%}")

        if "ragas_results" in st.session_state:
            res = st.session_state.ragas_results
            
            # Grid de métricas
            m1, m2, m3 = st.columns(3)
            
            # Sacamos los valores del objeto Result de RAGAS
            try:
                f_val = res["mean_scores"]["faithfulness"]
                r_val = res["mean_scores"]["answer_relevancy"]
                p_val = res["mean_scores"]["context_precision"]
            except:
                f_val = 0.0
                r_val = 0.0
                p_val = 0.0

            m1.metric("Faithfulness (Promedio)", f"{f_val:.2%}")
            m2.metric("Answer Relevancy", f"{r_val:.2%}")
            m3.metric("Context Precision", f"{p_val:.2%}")
            
            st.divider()
            
            # Desglose Individual
            st.subheader("📋 Desglose por Interacción")
            if "individual_samples" in res:
                for i, sample in enumerate(res["individual_samples"]):
                    col_idx, col_q, col_f, col_btn = st.columns([0.5, 6, 2, 1.5])
                    col_idx.markdown(f"**#{i+1}**")
                    col_q.markdown(f"_{sample['question'][:100]}..._")
                    
                    f_score = sample.get('faithfulness', 0)
                    risk_icon = "✅" if f_score > 0.8 else ("⚠️" if f_score > 0.5 else "🚨")
                    col_f.markdown(f"{risk_icon} **{f_score:.0%}**")
                    
                    if col_btn.button("🔎 Ver", key=f"btn_ragas_{i}"):
                        show_sample_details(sample)
                
            st.divider()
            
            # Botón de descarga JSON (Compliance)
            st.download_button(
                label="💾 Descargar Reporte RAGAS Completo (JSON)",
                data=json.dumps(res, indent=2, ensure_ascii=False),
                file_name=f"ragas_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("Configura la cantidad de muestras y ejecuta la evaluación para ver detalles.")

    # --- TAB 4: CENTRO DE COMANDO (OBSERVABILIDAD) ---
    with tab_command:
        st.markdown('<h1 class="main-header">Centro de Comando</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: {text_secondary}; margin-top: -15px;">Observabilidad en tiempo real: Servicios, Latencia y Application Insights.</p>', unsafe_allow_html=True)
        
        # --- SECCIÓN 1: HEARTBEAT DE SERVICIOS ---
        st.subheader("🫀 Estado de Servicios Azure (Heartbeat)")
        
        if st.button("🔄 Verificar Salud de Servicios", use_container_width=True, type="primary"):
            with st.spinner("Consultando servicios de Azure..."):
                from src.telemetry import check_azure_health
                health = check_azure_health()
                st.session_state.azure_health = health
        
        if "azure_health" in st.session_state:
            health = st.session_state.azure_health
            cols = st.columns(len(health))
            for i, (service, data) in enumerate(health.items()):
                with cols[i]:
                    status_color = "#10B981" if data["status"] == "healthy" else ("#F59E0B" if "no config" in data["status"] else "#EF4444")
                    st.markdown(f"""
                        <div style="text-align: center; padding: 15px; border-radius: 12px; border: 1px solid {border_color}; background: {bg_chat};">
                            <div style="font-size: 2rem;">{data['icon']}</div>
                            <div style="font-weight: 700; font-size: 0.85rem; color: {color_text}; margin-top: 5px;">{service}</div>
                            <div style="font-size: 0.75rem; color: {status_color}; font-weight: 600; margin-top: 3px;">{data['status'].upper()}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- SECCIÓN 2: TIMELINE DE TELEMETRÍA ---
        st.subheader("⏱️ Timeline de Latencia del Grafo")
        st.markdown(f'<p style="color: {text_secondary}; font-size: 0.85rem;">Historial de latencia por nodo de las últimas consultas procesadas.</p>', unsafe_allow_html=True)
        
        # Leer datos de telemetría del audit_log
        log_path = "outputs/governance/audit_log.jsonl"
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # Extraemos las últimas 10 entradas con telemetría
                timeline_data = []
                for line in reversed(lines[-20:]):
                    entry = json.loads(line)
                    meta = entry.get("metadata", {})
                    tele = meta.get("telemetry", {})
                    if tele and tele.get("nodes"):
                        timeline_data.append({
                            "query": entry.get("query", "N/A")[:50],
                            "total_ms": tele.get("total_ms", 0),
                            "nodes": tele.get("nodes", {}),
                            "timestamp": entry.get("timestamp", "")
                        })
                    if len(timeline_data) >= 10:
                        break
                
                if timeline_data:
                    for i, item in enumerate(timeline_data):
                        query_label = item['query']
                        total = item['total_ms']
                        nodes = item['nodes']
                        
                        with st.expander(f"#{i+1} — _{query_label}..._ ({total}ms)", expanded=(i == 0)):
                            node_cols = st.columns(len(nodes))
                            for j, (node_name, ms) in enumerate(nodes.items()):
                                pct = min(int((ms / max(total, 1)) * 100), 100)
                                icon = {"router": "🧠", "retriever": "🔍", "grader": "⚖️", "generator": "✍️"}.get(node_name, "⚙️")
                                with node_cols[j]:
                                    st.metric(f"{icon} {node_name.capitalize()}", f"{ms}ms")
                                    st.progress(pct / 100)
                else:
                    st.info("Todavía no hay datos de telemetría. Haz una consulta al chat para generar la primera medición.")
            except Exception as e:
                st.warning(f"No se pudo leer la telemetría: {e}")
        else:
            st.info("El archivo de auditoría aún no existe. Haz tu primera consulta para generar datos.")
        
        st.divider()
        
        # --- SECCIÓN 3: APPLICATION INSIGHTS ---
        st.subheader("🔵 Azure Application Insights")
        
        insights_key = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if insights_key:
            st.success("✅ **Conectado**. Los eventos de telemetría se envían a Application Insights en tiempo real.")
            st.markdown(f"""
                <div style="background: {bg_chat}; border: 1px solid {border_color}; border-left: 4px solid #3B82F6; padding: 15px; border-radius: 10px;">
                    <p style="color: {color_text}; margin: 0; font-size: 0.9rem;">
                        📡 Cada consulta al agente genera un evento <code>LegalGuard.GraphExecution</code> con la latencia por nodo.<br>
                        🔍 Puedes visualizarlos en <b>Azure Portal → Application Insights → Transaction Search</b>.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Application Insights no configurado. Añade `APPLICATIONINSIGHTS_CONNECTION_STRING` a las variables de entorno para activarlo.")
            st.markdown(f"""
                <div style="background: {bg_chat}; border: 1px solid {border_color}; padding: 15px; border-radius: 10px; font-size: 0.85rem; color: {text_secondary};">
                    💡 <b>Para activar:</b><br>
                    1. Crea un recurso Application Insights en Azure Portal.<br>
                    2. Copia la <b>Connection String</b>.<br>
                    3. Añádela como <code>APPLICATIONINSIGHTS_CONNECTION_STRING</code> en los secrets de Azure Container Apps.
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- SECCIÓN DE AUDITORÍA Y COMPLIANCE ---
    with st.expander("🛡️ Centro de Trazabilidad y Compliance"):
        col_log1, col_log2 = st.columns([2, 1])
        with col_log1:
            st.markdown("""
                **Log de Auditoría (Immutable Audit Trail)**
                Captura cada consulta, respuesta y fragmento utilizado, incluyendo marcas de tiempo y hashes de integridad.
            """)
        with col_log2:
            log_path = "outputs/governance/audit_log.jsonl"
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    audit_data = f.read()
                st.download_button(
                    label="📂 Descargar Log Completo (JSONL)",
                    data=audit_data,
                    file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.jsonl",
                    mime="application/jsonl",
                    use_container_width=True
                )
            else:
                st.error("Archivo de auditoría no encontrado.")

    if not (st.session_state.md_content or selected_docs):
        st.info("⬅️ Sube un PDF o selecciona documentos en el **Filtro de Memoria** para comenzar.")
    else:
        st.success("✅ Sistema listo para consultas legales.")

