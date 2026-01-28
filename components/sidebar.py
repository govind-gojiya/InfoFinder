"""Sidebar component for chat history and navigation."""

import streamlit as st
from datetime import datetime

from models.schemas import Chat
from services.chat_manager import ChatManager
from services.auth import auth_service
from utils.helpers import format_timestamp, truncate_text


def render_sidebar(chat_manager: ChatManager):
    """
    Render the sidebar with chat history.
    
    Args:
        chat_manager: ChatManager instance
    """
    with st.sidebar:
        # App branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.8rem; margin: 0; color: #8b5cf6;">
                📚 Info Finder
            </h1>
            <p style="color: #8b949e; font-size: 0.85rem; margin-top: 0.3rem;">
                Chat with your documents
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # User info section
        _render_user_info()
        
        st.divider()
        
        # New chat button
        if st.button(
            "➕ New Chat",
            key="new_chat_btn",
            use_container_width=True,
            type="primary"
        ):
            new_chat = chat_manager.create_chat("New Chat")
            st.session_state.current_chat_id = new_chat.id
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Search chats
        search_query = st.text_input(
            "🔍 Search chats",
            placeholder="Search by title...",
            label_visibility="collapsed"
        )
        
        # Get chats
        if search_query:
            chats = chat_manager.search_chats(search_query)
        else:
            chats = chat_manager.list_chats()
        
        # Group chats by time
        today_chats = []
        yesterday_chats = []
        older_chats = []
        
        now = datetime.now()
        for chat in chats:
            diff = now - chat.updated_at
            if diff.days == 0:
                today_chats.append(chat)
            elif diff.days == 1:
                yesterday_chats.append(chat)
            else:
                older_chats.append(chat)
        
        # Render chat groups
        if today_chats:
            st.markdown("**Today**")
            _render_chat_list(today_chats, chat_manager)
        
        if yesterday_chats:
            st.markdown("**Yesterday**")
            _render_chat_list(yesterday_chats, chat_manager)
        
        if older_chats:
            st.markdown("**Previous**")
            _render_chat_list(older_chats, chat_manager)
        
        if not chats:
            st.markdown(
                "<p style='color: #8b949e; text-align: center; padding: 2rem 0;'>"
                "No chats yet.<br>Start a new conversation!</p>",
                unsafe_allow_html=True
            )
        
        # Footer
        st.divider()
        _render_settings()


def _render_user_info():
    """Render user info and logout button."""
    user_name = st.session_state.get("user_name", "User")
    user_email = st.session_state.get("user_email", "")
    
    col1, col2 = st.columns([0.75, 0.25])
    
    with col1:
        st.markdown(f"""
        <div style="padding: 0.5rem 0;">
            <div style="color: #e6edf3; font-weight: 600;">👤 {user_name}</div>
            <div style="color: #8b949e; font-size: 0.75rem;">{user_email}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚪", key="logout_btn", help="Logout"):
            _logout()


def _logout():
    """Handle user logout."""
    from components.auth import logout
    logout()


def _render_chat_list(chats: list[Chat], chat_manager: ChatManager):
    """Render a list of chat items."""
    for chat in chats:
        is_selected = st.session_state.get("current_chat_id") == chat.id
        
        col1, col2 = st.columns([0.85, 0.15])
        
        with col1:
            # Chat button
            button_label = truncate_text(chat.title, 28)
            if chat.document_count > 0:
                button_label = f"📎 {button_label}"
            
            if st.button(
                button_label,
                key=f"chat_{chat.id}",
                use_container_width=True,
                type="secondary" if not is_selected else "primary"
            ):
                st.session_state.current_chat_id = chat.id
                # Load messages from chat
                loaded_chat = chat_manager.get_chat(chat.id)
                if loaded_chat:
                    st.session_state.messages = [
                        {"role": m.role.value, "content": m.content}
                        for m in loaded_chat.messages
                    ]
                st.rerun()
        
        with col2:
            # Delete button
            if st.button(
                "🗑️",
                key=f"delete_{chat.id}",
                help="Delete chat"
            ):
                chat_manager.delete_chat(chat.id)
                if st.session_state.get("current_chat_id") == chat.id:
                    st.session_state.current_chat_id = None
                    st.session_state.messages = []
                st.rerun()


def _render_settings():
    """Render settings section in sidebar."""
    with st.expander("⚙️ Settings"):
        # LLM Provider selection
        provider = st.selectbox(
            "LLM Provider",
            options=["groq", "ollama"],
            index=0 if st.session_state.get("llm_provider", "groq") == "groq" else 1,
            key="llm_provider_select"
        )
        st.session_state.llm_provider = provider
        
        if provider == "groq":
            # Get current API key from session
            current_key = st.session_state.get("groq_api_key", "")
            
            # Show masked key info
            if current_key:
                masked_key = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "***"
                st.markdown(f"<small style='color: #22c55e;'>✓ Key configured: {masked_key}</small>", unsafe_allow_html=True)
            
            api_key = st.text_input(
                "Update Groq API Key",
                type="password",
                value="",
                placeholder="Enter new key to update...",
                help="Your Groq API key for LLM access"
            )
            
            # Update key if provided
            if api_key and api_key.strip():
                user_id = st.session_state.get("user_id")
                if user_id:
                    success, error = auth_service.update_groq_key(user_id, api_key)
                    if success:
                        st.session_state.groq_api_key = api_key.strip()
                        st.success("API key updated!", icon="✅")
                        st.rerun()
                    else:
                        st.error(error)
            
            st.markdown(
                "<small>[Get free Groq API key →](https://console.groq.com/keys)</small>",
                unsafe_allow_html=True
            )
        else:
            # Ollama settings (backup/offline option)
            st.markdown("<small style='color: #f59e0b;'>⚠️ Ollama is for local/offline use only</small>", unsafe_allow_html=True)
            
            ollama_url = st.text_input(
                "Ollama URL",
                value=st.session_state.get("ollama_url", "http://localhost:11434"),
                help="URL of your Ollama server"
            )
            st.session_state.ollama_url = ollama_url
            
            ollama_model = st.text_input(
                "Ollama Model",
                value=st.session_state.get("ollama_model", "llama3.1"),
                help="Model name (e.g., llama3.1, mistral)"
            )
            st.session_state.ollama_model = ollama_model
        
        # Search settings
        st.markdown("---")
        st.markdown("**Search Settings**")
        
        top_k = st.slider(
            "Results to retrieve",
            min_value=3,
            max_value=20,
            value=st.session_state.get("top_k", 10),
            help="Number of document chunks to retrieve"
        )
        st.session_state.top_k = top_k
        
        use_reranking = st.checkbox(
            "Enable reranking",
            value=st.session_state.get("use_reranking", True),
            help="Use cross-encoder for better relevance"
        )
        st.session_state.use_reranking = use_reranking
