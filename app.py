"""
Info Finder - RAG-powered PDF Chat Application

A Streamlit application for chatting with PDF documents using:
- Hybrid search (vector + keyword with RRF fusion)
- Cross-encoder reranking
- Free LLM providers (Groq / Ollama)
- User authentication with SQLite
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

import config
from models.schemas import Chat, MessageRole
from services.chat_manager import ChatManager
from services.hybrid_search import HybridSearchService
from services.reranker import RerankerService
from services.llm import LLMService
from services.pdf_processor import PDFProcessor
from services.chunker import DocumentChunker
from services.embeddings import EmbeddingService
from components.sidebar import render_sidebar
from components.chat import render_chat_interface
from components.auth import render_auth_page, require_auth


# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* ========== DARK THEME ========== */
    
    /* Main app background */
    .stApp {
        background: #0d1117 !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background: #0d1117 !important;
    }
    
    /* All text should be light */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #e6edf3 !important;
    }
    
    /* Headings */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
    }
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
    }
    
    [data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #e6edf3 !important;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        transition: all 0.2s;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: #30363d !important;
        border-color: #8b5cf6 !important;
    }
    
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    }
    
    /* Sidebar inputs */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] .stTextInput input {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
    }
    
    /* ========== CHAT MESSAGES ========== */
    [data-testid="stChatMessage"] {
        background: #161b22 !important;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #30363d !important;
    }
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li {
        color: #e6edf3 !important;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #1c2128 !important;
        border-left: 4px solid #8b5cf6 !important;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #161b22 !important;
        border-left: 4px solid #22c55e !important;
    }
    
    /* Chat message text */
    .stChatMessage .stMarkdown,
    .stChatMessage .stMarkdown p,
    .stChatMessage .stMarkdown li,
    .stChatMessage .stMarkdown span {
        color: #e6edf3 !important;
        line-height: 1.6;
    }
    
    .stChatMessage .stMarkdown strong {
        color: #ffffff !important;
    }
    
    .stChatMessage .stMarkdown code {
        background: #30363d !important;
        color: #79c0ff !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    .stChatMessage .stMarkdown pre {
        background: #0d1117 !important;
        color: #e6edf3 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363d !important;
    }
    
    /* ========== INPUTS ========== */
    .stTextInput input,
    .stTextArea textarea {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 8px;
        color: #e6edf3 !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
    }
    
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #8b949e !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    
    [data-testid="stChatInput"] textarea {
        background: #21262d !important;
        color: #e6edf3 !important;
    }
    
    /* ========== FILE UPLOADER ========== */
    [data-testid="stFileUploader"] {
        background: #161b22 !important;
        border: 2px dashed #30363d !important;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6 !important;
        background: #1c2128 !important;
    }
    
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: #e6edf3 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: #21262d !important;
        border: 1px solid #30363d !important;
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        border-radius: 8px;
        color: #e6edf3 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #21262d !important;
    }
    
    .streamlit-expanderContent {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
    }
    
    [data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px;
    }
    
    /* ========== BUTTONS ========== */
    .stButton button {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: #30363d !important;
        border-color: #8b5cf6 !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        border: none !important;
        color: white !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    }
    
    /* ========== SELECT BOX ========== */
    .stSelectbox > div > div {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background: #21262d !important;
    }
    
    .stSelectbox [data-baseweb="select"] * {
        color: #e6edf3 !important;
    }
    
    /* ========== SLIDER ========== */
    .stSlider > div > div {
        color: #e6edf3 !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: #30363d !important;
    }
    
    /* ========== CHECKBOX ========== */
    .stCheckbox label span {
        color: #e6edf3 !important;
    }
    
    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
    }
    
    .stProgress > div {
        background: #30363d !important;
    }
    
    /* ========== ALERTS ========== */
    .stAlert {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
    }
    
    .stSuccess {
        background: #0d1f0d !important;
        border-color: #22c55e !important;
    }
    
    .stError {
        background: #1f0d0d !important;
        border-color: #ef4444 !important;
    }
    
    .stWarning {
        background: #1f1a0d !important;
        border-color: #f59e0b !important;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22 !important;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #30363d !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
    }
    
    /* ========== FORM ========== */
    [data-testid="stForm"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* ========== DIVIDERS ========== */
    hr {
        border-color: #30363d !important;
    }
    
    /* ========== HIDE STREAMLIT BRANDING ========== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #161b22;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    
    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-color: #8b5cf6 !important;
    }
    
    /* ========== MARKDOWN LINKS ========== */
    .stMarkdown a {
        color: #58a6ff !important;
    }
    
    .stMarkdown a:hover {
        color: #79c0ff !important;
    }
    
    /* ========== TOOLTIP ========== */
    [data-baseweb="tooltip"] {
        background: #21262d !important;
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = config.LLM_PROVIDER
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = config.TOP_K_RETRIEVAL
    
    if "use_reranking" not in st.session_state:
        st.session_state.use_reranking = True
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False


@st.cache_resource
def get_embedding_service():
    """Get cached embedding service."""
    return EmbeddingService()


@st.cache_resource
def get_reranker_service():
    """Get cached reranker service."""
    return RerankerService()


def get_llm_service():
    """Get LLM service based on current settings."""
    provider = st.session_state.get("llm_provider", "groq")
    
    # Set environment variables from session state
    if provider == "groq":
        api_key = st.session_state.get("groq_api_key", "")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        ollama_url = st.session_state.get("ollama_url", config.OLLAMA_BASE_URL)
        ollama_model = st.session_state.get("ollama_model", config.OLLAMA_MODEL)
        os.environ["OLLAMA_BASE_URL"] = ollama_url
        os.environ["OLLAMA_MODEL"] = ollama_model
    
    return LLMService(provider=provider)


def get_hybrid_search(chat_id: str) -> HybridSearchService:
    """Get hybrid search service for a chat."""
    embedding_service = get_embedding_service()
    return HybridSearchService(chat_id, embedding_service)


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Check authentication
    if not require_auth():
        # Show login/signup page
        render_auth_page()
        return
    
    # Get current user ID
    user_id = st.session_state.get("user_id")
    if not user_id:
        # Invalid state, reset auth
        st.session_state.authenticated = False
        st.rerun()
    
    # Initialize services with user ID
    chat_manager = ChatManager(user_id)
    pdf_processor = PDFProcessor()
    chunker = DocumentChunker()
    reranker = get_reranker_service()
    
    # Render sidebar
    render_sidebar(chat_manager)
    
    # Get current chat
    current_chat = None
    hybrid_search = None
    
    if st.session_state.current_chat_id:
        current_chat = chat_manager.get_chat(st.session_state.current_chat_id)
        if current_chat:
            hybrid_search = get_hybrid_search(current_chat.id)
        else:
            # Chat not found (or not owned by user), reset selection
            st.session_state.current_chat_id = None
            st.session_state.messages = []
    
    # Get LLM service
    try:
        llm_service = get_llm_service()
    except ValueError as e:
        llm_service = None
        if current_chat:
            st.warning(
                "⚠️ LLM not configured. Please set your API key in the sidebar settings."
            )
    
    # Render main chat interface
    render_chat_interface(
        chat=current_chat,
        chat_manager=chat_manager,
        hybrid_search=hybrid_search,
        reranker=reranker,
        llm_service=llm_service,
        pdf_processor=pdf_processor,
        chunker=chunker
    )


if __name__ == "__main__":
    main()
