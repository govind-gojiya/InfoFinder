"""Chat interface component."""

import streamlit as st
from typing import Optional

import config
from models.schemas import Chat, SearchResult, MessageRole
from services.chat_manager import ChatManager
from services.hybrid_search import HybridSearchService
from services.reranker import RerankerService
from services.llm import LLMService
from services.pdf_processor import PDFProcessor
from services.chunker import DocumentChunker
from utils.helpers import format_file_size


def render_chat_interface(
    chat: Optional[Chat],
    chat_manager: ChatManager,
    hybrid_search: Optional[HybridSearchService],
    reranker: RerankerService,
    llm_service: LLMService,
    pdf_processor: PDFProcessor,
    chunker: DocumentChunker
):
    """
    Render the main chat interface.
    
    Args:
        chat: Current chat object
        chat_manager: ChatManager instance
        hybrid_search: HybridSearchService instance
        reranker: RerankerService instance
        llm_service: LLMService instance
        pdf_processor: PDFProcessor instance
        chunker: DocumentChunker instance
    """
    if not chat:
        _render_welcome_screen()
        return
    
    # Header with chat info
    _render_chat_header(chat, hybrid_search)
    
    # File upload section
    _render_file_upload(
        chat, chat_manager, hybrid_search,
        pdf_processor, chunker
    )
    
    # Chat messages
    _render_messages()
    
    # Chat input
    _render_chat_input(
        chat, chat_manager, hybrid_search,
        reranker, llm_service
    )


def _render_welcome_screen():
    """Render welcome screen when no chat is selected."""
    st.markdown("""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
        <h1 style="color: #8b5cf6; margin-bottom: 0.5rem;">Welcome to Info Finder</h1>
        <p style="color: #8b949e; max-width: 500px; line-height: 1.6;">
            Upload your PDF documents and chat with them using AI.
            Start by creating a new chat from the sidebar.
        </p>
        <div style="margin-top: 2rem; padding: 1.5rem; background: #161b22; border: 1px solid #30363d; border-radius: 12px; max-width: 400px;">
            <h3 style="color: #e6edf3; margin-bottom: 1rem;">✨ Features</h3>
            <ul style="text-align: left; color: #8b949e; line-height: 1.8;">
                <li>📄 Extract text, tables, and images from PDFs</li>
                <li>🔍 Hybrid search (semantic + keyword)</li>
                <li>🎯 Intelligent reranking for better results</li>
                <li>💬 Natural conversation with your documents</li>
                <li>📝 Chat history preservation</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_chat_header(chat: Chat, hybrid_search: Optional[HybridSearchService]):
    """Render chat header with title and document count."""
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.markdown(f"### 💬 {chat.title}")
    
    with col2:
        doc_count = hybrid_search.document_count() if hybrid_search else 0
        if doc_count > 0:
            st.markdown(
                f"<div style='text-align: right; color: #8b5cf6; padding-top: 0.5rem;'>"
                f"📎 {doc_count} document chunks indexed</div>",
                unsafe_allow_html=True
            )


def _render_file_upload(
    chat: Chat,
    chat_manager: ChatManager,
    hybrid_search: Optional[HybridSearchService],
    pdf_processor: PDFProcessor,
    chunker: DocumentChunker
):
    """Render file upload section."""
    with st.expander("📤 Upload Documents", expanded=not bool(st.session_state.get("messages"))):
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"file_upload_{chat.id}",
            help="Upload one or more PDF files to add to this chat"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                _process_uploaded_files(
                    uploaded_files, chat, chat_manager,
                    hybrid_search, pdf_processor, chunker
                )


def _process_uploaded_files(
    uploaded_files,
    chat: Chat,
    chat_manager: ChatManager,
    hybrid_search: HybridSearchService,
    pdf_processor: PDFProcessor,
    chunker: DocumentChunker
):
    """Process uploaded PDF files."""
    total_docs = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        progress = (idx) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {uploaded_file.name}...")
        
        try:
            # Extract content from PDF
            documents = pdf_processor.process_uploaded_file(uploaded_file)
            
            # Chunk documents
            chunked_docs = chunker.chunk_documents(documents)
            
            # Add to search index
            if hybrid_search:
                hybrid_search.add_documents(chunked_docs)
                total_docs += len(chunked_docs)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Update chat document count
    if hybrid_search:
        chat_manager.update_document_count(chat.id, hybrid_search.document_count())
    
    st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_docs} chunks!")
    st.rerun()


def _render_messages():
    """Render chat messages."""
    messages = st.session_state.get("messages", [])
    
    for message in messages:
        render_message(message["role"], message["content"])


def render_message(role: str, content: str, sources: list[SearchResult] = None):
    """
    Render a single chat message.
    
    Args:
        role: Message role ("user" or "assistant")
        content: Message content
        sources: Optional list of source documents
    """
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            # Show sources if available
            if sources:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(sources, 1):
                        st.markdown(
                            f"**Source {i}** - {source.document.source_file} "
                            f"(Page {source.document.page_number})"
                        )
                        st.markdown(
                            f"<div style='background: #21262d; padding: 0.75rem; "
                            f"border-radius: 6px; font-size: 0.85rem; color: #e6edf3; "
                            f"border: 1px solid #30363d; line-height: 1.5;'>"
                            f"{source.document.content[:300]}...</div>",
                            unsafe_allow_html=True
                        )


def _render_chat_input(
    chat: Chat,
    chat_manager: ChatManager,
    hybrid_search: Optional[HybridSearchService],
    reranker: RerankerService,
    llm_service: LLMService
):
    """Render chat input and handle message submission."""
    # Check if we have documents
    has_documents = hybrid_search and hybrid_search.document_count() > 0
    
    placeholder = (
        "Ask a question about your documents..."
        if has_documents
        else "Upload documents first to start chatting..."
    )
    
    if prompt := st.chat_input(placeholder, disabled=not has_documents):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_manager.add_message(chat.id, MessageRole.USER, prompt)
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            try:
                status_placeholder = st.empty()
                top_k = st.session_state.get("top_k", 10)
                use_multi_query = st.session_state.get("use_multi_query", True)
                similar_queries = [prompt]  # Default: just the original query
                
                if use_multi_query:
                    # Step 1: Generate similar queries for multi-query retrieval
                    status_placeholder.markdown("🔍 *Generating search queries...*")
                    similar_queries = llm_service.generate_similar_queries(prompt, num_queries=5)
                    
                    # Step 2: Multi-query search with RRF fusion
                    status_placeholder.markdown(f"🔎 *Searching with {len(similar_queries)} queries...*")
                    search_results = hybrid_search.multi_query_search(
                        similar_queries, 
                        top_k=top_k
                    )
                else:
                    # Standard single-query hybrid search
                    status_placeholder.markdown("🔎 *Searching documents...*")
                    search_results = hybrid_search.search(prompt, top_k=top_k)
                
                # Step 3: Rerank the final results if enabled
                if st.session_state.get("use_reranking", True) and search_results:
                    status_placeholder.markdown("📊 *Reranking results...*")
                    search_results = reranker.rerank(
                        prompt, search_results, top_k=config.TOP_K_RERANK
                    )
                
                # Clear status
                status_placeholder.empty()
                
                # Show the queries used (collapsed) - only if multi-query was used
                if use_multi_query and len(similar_queries) > 1:
                    with st.expander("🔍 Search Queries Used", expanded=False):
                        for i, q in enumerate(similar_queries, 1):
                            prefix = "**Original:**" if i == 1 else f"**Query {i}:**"
                            st.markdown(f"{prefix} {q}")
                
                # Format context
                context = llm_service.format_context(search_results)
                
                # Get conversation history
                conversation_history = chat.get_conversation_history(max_messages=6)
                
                # Generate response with streaming
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in llm_service.generate_response_stream(
                    question=prompt,
                    context=context,
                    conversation_history=conversation_history
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Show sources
                if search_results:
                    with st.expander("📚 Sources", expanded=False):
                        for i, result in enumerate(search_results, 1):
                            st.markdown(
                                f"**Source {i}** - {result.document.source_file} "
                                f"(Page {result.document.page_number})"
                            )
                            st.markdown(
                                f"<div style='background: #21262d; padding: 0.75rem; "
                                f"border-radius: 6px; font-size: 0.85rem; margin-bottom: 0.5rem; "
                                f"color: #e6edf3; border: 1px solid #30363d; line-height: 1.5;'>"
                                f"{result.document.content[:300]}...</div>",
                                unsafe_allow_html=True
                            )
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                chat_manager.add_message(
                    chat.id, MessageRole.ASSISTANT,
                    full_response, search_results
                )
                
                # Update chat title if this is the first message
                if len(st.session_state.messages) == 2:
                    try:
                        new_title = llm_service.generate_chat_title(prompt)
                        chat_manager.update_chat_title(chat.id, new_title)
                    except Exception:
                        pass
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}"
                })

