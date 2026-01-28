# 📚 Info Finder

A powerful RAG (Retrieval-Augmented Generation) application built with Streamlit for chatting with your PDF documents. Features user authentication, hybrid search, intelligent reranking, and free LLM support.

![Info Finder](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **🔐 User Authentication**: Secure login/signup with SQLite database
- **👤 User-Specific Data**: Each user has their own isolated chats and documents
- **🔑 Personal API Keys**: Users provide their own Groq API key (free tier available)
- **📄 PDF Processing**: Extract text, tables, and images from PDF documents
- **🔍 Hybrid Search**: Combines semantic (vector) search with keyword (BM25) search
- **🎯 RRF Fusion**: Reciprocal Rank Fusion for optimal result combination
- **📊 Cross-Encoder Reranking**: Improved relevance with neural reranking
- **💬 Chat History**: Persistent chat sessions like ChatGPT
- **🆓 Free LLM Support**: Works with Groq (free cloud) or Ollama (free local backup)
- **🌙 Dark Theme**: Beautiful, modern dark UI

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd info_finder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Your Free Groq API Key

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up for a free account
3. Create a new API key (starts with `gsk_`)
4. Save it - you'll need it when signing up!

### 3. Run the Application

```bash
streamlit run app.py --server.headless true
```

The app will open at `http://localhost:8501`

### 4. Create Your Account

1. Click **Sign Up** on the authentication page
2. Enter your name, email, and password
3. Paste your **Groq API Key** (required)
4. Click **Create Account**

That's it! You're ready to start chatting with your documents.

## 📖 Usage

1. **Login/Sign Up**: Create an account or login with existing credentials
2. **Create a New Chat**: Click "➕ New Chat" in the sidebar
3. **Upload PDFs**: Expand the "📤 Upload Documents" section and upload your PDF files
4. **Process Files**: Click "🚀 Process Files" to index your documents
5. **Ask Questions**: Type your questions in the chat input
6. **View Sources**: Click "📚 Sources" to see which documents were used

## 🔐 Authentication System

### User Registration
- **Required fields**: Name, Email, Password, Groq API Key
- **Email normalization**: Prevents duplicate accounts (removes + aliases, lowercase)
- **Password security**: Hashed with SHA-256 + salt
- **API key validation**: Must start with `gsk_`

### User Isolation
- Each user can only see their own chats
- Documents are isolated per user
- API keys are stored securely per user

### Settings
- Update your Groq API key anytime
- Switch to Ollama for offline use (backup option)
- Adjust search settings (results count, reranking)

## 🏗️ Architecture

```
info_finder/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── components/
│   ├── auth.py              # Login/Signup UI
│   ├── sidebar.py           # Sidebar with chat history
│   └── chat.py              # Chat interface component
├── services/
│   ├── database.py          # SQLite database for users
│   ├── auth.py              # Authentication service
│   ├── pdf_processor.py     # PDF text/table/image extraction
│   ├── chunker.py           # Document chunking
│   ├── embeddings.py        # Sentence-transformer embeddings
│   ├── vector_store.py      # ChromaDB vector store
│   ├── keyword_search.py    # BM25 keyword search
│   ├── hybrid_search.py     # Hybrid search with RRF
│   ├── reranker.py          # Cross-encoder reranking
│   ├── llm.py               # LLM providers (Groq/Ollama)
│   └── chat_manager.py      # Chat session management (per user)
├── models/
│   └── schemas.py           # Data models
├── utils/
│   └── helpers.py           # Utility functions
└── data/
    ├── info_finder.db       # SQLite database (auto-created)
    ├── chats/               # Chat JSON files
    └── vector_stores/       # ChromaDB stores
```

## 🔧 Configuration

Edit `config.py` or use environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Maximum chunk size for documents |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 20 | Initial retrieval count |
| `TOP_K_RERANK` | 5 | Results after reranking |
| `RRF_K` | 60 | RRF constant |

## 🔍 How It Works

### 1. Authentication
- Users register with email, password, and Groq API key
- Email is normalized (lowercase, + aliases removed)
- Each user gets isolated storage for chats and documents

### 2. Document Processing
- PDFs are processed using PyMuPDF (text/images) and pdfplumber (tables)
- Content is chunked using recursive character splitting

### 3. Indexing
- Text chunks are embedded using `all-MiniLM-L6-v2`
- Stored in ChromaDB for vector search
- Indexed with BM25 for keyword search

### 4. Retrieval
- **Vector Search**: Semantic similarity using embeddings
- **Keyword Search**: BM25 algorithm for exact matches
- **RRF Fusion**: Combines both rankings optimally

### 5. Reranking
- Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores query-document pairs
- Top results are reranked for better relevance

### 6. Generation
- Context + conversation history sent to LLM
- Uses **user's personal Groq API key** for all requests
- Streaming response for better UX

## 🔑 Why Personal API Keys?

Each user provides their own Groq API key because:
- **Free tier**: Groq offers generous free limits
- **Scalability**: No server-side API costs as users grow
- **Control**: Users manage their own usage and limits
- **Privacy**: Direct connection between user and LLM provider

## 🛠️ Technologies

- **Frontend**: Streamlit
- **Database**: SQLite (users)
- **PDF Processing**: PyMuPDF, pdfplumber
- **Embeddings**: sentence-transformers
- **Vector Store**: ChromaDB
- **Keyword Search**: rank-bm25
- **Reranking**: Cross-encoder
- **LLM**: Groq (primary) / Ollama (backup)

## 🔒 Security Notes

- Passwords are hashed (never stored in plain text)
- API keys are stored in database (consider encryption for production)
- Email normalization prevents account spoofing
- User isolation ensures data privacy

## 📝 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Groq](https://groq.com/) for free LLM API
- [Ollama](https://ollama.ai/) for local LLM support
- [sentence-transformers](https://www.sbert.net/) for embeddings
