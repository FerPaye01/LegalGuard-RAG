import streamlit as st
import httpx
import json
import asyncio
from typing import Generator

# 1. Configuración de página (SIEMPRE PRIMERO)
st.set_page_config(
    page_title="LegalGuard RAG | Smart Contract Review",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Estado de Sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

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

# 5. Estilos Personalizados (God Mode UI)
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

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
    }}
    [data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}
    
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

# 7. Sidebar Content
with st.sidebar:
    st.markdown('<h1 style="font-size: 3.5rem; margin-bottom: -20px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🏛️</h1>', unsafe_allow_html=True)
    st.title("LegalGuard RAG")
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Enterprise-Grade Legal AI</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
    
    st.subheader("Auditoría de Riesgo")
    st.progress(65)
    st.markdown(f"<p style='font-size: 0.8rem; margin-top: -10px; color: #F59E0B; font-weight: bold;'>Risk Score: 65/100 (Medio)</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B; padding: 10px; border-radius: 4px;">
            <span style="color: #F59E0B; font-weight: 600;">⚠️ 12 Cláusulas críticas</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.selectbox("Documento Activo:", ["Contrato_Maestro_CUAD.pdf", "SOP_Clinico_Emergencias.pdf"])
    
    if st.button("🗑️ Limpiar Sesión", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 8. Main Layout
col_pdf, col_chat = st.columns([4.2, 5.8])

with col_pdf:
    st.markdown(f'<h3 style="color: {color_text}; font-weight: 700; display: flex; align-items: center; gap: 8px;">📄 Documento Fuente</h3>', unsafe_allow_html=True)
    bg_info = "#1E293B" if st.session_state.dark_mode else "#E0F2FE"
    color_info = "#3B82F6" if st.session_state.dark_mode else "#0369A1"
    st.markdown(f"""
        <div style="background-color: {bg_info}; border-left: 4px solid #3B82F6; padding: 12px; border-radius: 8px; font-size: 0.9rem; color: {color_info}; margin-bottom: 20px;">
            💡 Selecciona un texto en el chat para resaltar su origen en el documento.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="pdf-viewer-placeholder">
            <div style="text-align: center;">
                <h4 style="margin: 0; color: {color_text};">Área de Visualización de PDF</h4>
                <p style="font-size: 0.9rem; margin-top: 5px; color: {text_secondary};">El motor XAI renderizará la evidencia aquí</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_chat:
    st.markdown('<h1 class="main-header">Asistente de Auditoría</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: {text_secondary}; margin-top: -15px;">Chat fundamentado en el documento activo con mitigación de alucinaciones.</p>', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("¿Qué deseas saber sobre este contrato?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            thought_container = st.status("LegalGuard está pensando...")
            message_placeholder = st.empty()
            stream_state = {"full_response": ""}
            
            async def run_chat():
                async for event in stream_from_backend(prompt):
                    if "error" in event:
                        st.error(event["error"])
                        break
                    if "status" in event:
                        thought_container.write(f"➜ **{event['node']}**: {event['status']}")
                    if "partial_answer" in event and event["partial_answer"]:
                        stream_state["full_response"] = event["partial_answer"]
                        message_placeholder.markdown(stream_state["full_response"] + "▌")
                
                message_placeholder.markdown(stream_state["full_response"])
                thought_container.update(label="Análisis de Auditoría Completo", state="complete")
                return stream_state["full_response"]

            final_text = asyncio.run(run_chat())
            st.session_state.messages.append({"role": "assistant", "content": final_text})
