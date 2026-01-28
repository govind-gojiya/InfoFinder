"""Authentication UI components for login and signup."""

import streamlit as st
from services.auth import auth_service, User


def render_auth_page():
    """
    Render the authentication page (login/signup).
    
    Returns:
        User object if authenticated, None otherwise
    """
    # Add custom CSS for auth page
    st.markdown("""
    <style>
        /* Auth page specific styles */
        .auth-container {
            max-width: 420px;
            margin: 0 auto;
            padding: 1.5rem;
        }
        
        .auth-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .auth-header h1 {
            color: #8b5cf6 !important;
            font-size: 2rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .auth-header p {
            color: #8b949e !important;
            font-size: 0.9rem !important;
            margin: 0 !important;
        }
        
        /* Tab styling fix */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0 !important;
            background: #21262d !important;
            border-radius: 10px !important;
            padding: 4px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            flex: 1 !important;
            justify-content: center !important;
            padding: 0.6rem 1rem !important;
            font-size: 0.9rem !important;
            border-radius: 8px !important;
            border: none !important;
            background: transparent !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem !important;
        }
        
        /* Compact form styling */
        .compact-form .stTextInput {
            margin-bottom: 0.5rem !important;
        }
        
        .compact-form .stTextInput label {
            font-size: 0.85rem !important;
            margin-bottom: 0.2rem !important;
        }
        
        .compact-form .stTextInput input {
            padding: 0.5rem 0.75rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Form container */
        [data-testid="stForm"] {
            padding: 1rem !important;
            border-radius: 10px !important;
        }
        
        /* Required field indicator */
        .required-label {
            color: #ef4444 !important;
            font-size: 0.75rem !important;
        }
        
        /* API key info box */
        .api-key-info {
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.5rem 0;
            font-size: 0.8rem;
        }
        
        .api-key-info a {
            color: #8b5cf6 !important;
            text-decoration: none;
        }
        
        .api-key-info a:hover {
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Center the auth form with tighter columns
    col1, col2, col3 = st.columns([1, 1.8, 1])
    
    with col2:
        # Compact header
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size: 2rem;">📚</span>
            <span style="color: #8b5cf6; font-size: 1.5rem; font-weight: 700; margin-left: 0.5rem;">Info Finder</span>
            <p style="color: #8b949e; font-size: 0.85rem; margin: 0.3rem 0 0 0;">Chat with your PDF documents using AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize auth mode in session state
        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "login"
        
        # Custom toggle buttons instead of tabs
        col_login, col_signup = st.columns(2)
        
        with col_login:
            if st.button(
                "🔐 Login",
                key="btn_login_tab",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "login" else "secondary"
            ):
                st.session_state.auth_mode = "login"
                st.rerun()
        
        with col_signup:
            if st.button(
                "📝 Sign Up",
                key="btn_signup_tab",
                use_container_width=True,
                type="primary" if st.session_state.auth_mode == "signup" else "secondary"
            ):
                st.session_state.auth_mode = "signup"
                st.rerun()
        
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        
        # Render appropriate form
        if st.session_state.auth_mode == "login":
            user = _render_login_form()
            if user:
                return user
        else:
            user = _render_signup_form()
            if user:
                return user
    
    return None


def _render_login_form() -> User | None:
    """Render compact login form."""
    with st.form("login_form", clear_on_submit=False):
        st.markdown("**Welcome Back!** Enter your credentials.")
        
        email = st.text_input(
            "Email",
            placeholder="your@email.com",
            key="login_email"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )
        
        submitted = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            if not email or not password:
                st.error("Please fill in all fields")
                return None
            
            user, error = auth_service.login(email, password)
            
            if user:
                # Check if user has Groq API key
                if not user.groq_api_key:
                    st.error("Your account doesn't have a Groq API key. Please contact support or re-register.")
                    return None
                
                st.success("Login successful!")
                st.session_state.user_id = user.id
                st.session_state.user_name = user.name
                st.session_state.user_email = user.email
                st.session_state.groq_api_key = user.groq_api_key
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error(error)
    
    return None


def _render_signup_form() -> User | None:
    """Render compact signup form with required Groq key."""
    with st.form("signup_form", clear_on_submit=False):
        st.markdown("**Create Account** to start chatting.")
        
        # Two columns for name and email
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name *", placeholder="John", key="signup_name")
        with col2:
            email = st.text_input("Email *", placeholder="you@email.com", key="signup_email")
        
        # Two columns for passwords
        col1, col2 = st.columns(2)
        with col1:
            password = st.text_input("Password *", type="password", placeholder="Min 6 chars", key="signup_password")
        with col2:
            confirm_password = st.text_input("Confirm *", type="password", placeholder="Confirm", key="signup_confirm_password")
        
        # Groq API Key - Required (compact info)
        st.markdown("""<div style="background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 0.5rem; margin: 0.3rem 0; font-size: 0.8rem;">
            🔑 <strong>Groq API Key (Required)</strong> - <a href="https://console.groq.com/keys" target="_blank" style="color: #8b5cf6;">Get free key</a>
        </div>""", unsafe_allow_html=True)
        
        groq_key = st.text_input("Groq API Key *", type="password", placeholder="gsk_...", key="signup_groq_key", label_visibility="collapsed")
        
        submitted = st.form_submit_button(
            "Create Account",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            # Validate all fields including Groq key
            if not name or not email or not password:
                st.error("Please fill in all required fields")
                return None
            
            if password != confirm_password:
                st.error("Passwords do not match")
                return None
            
            if not groq_key or not groq_key.strip():
                st.error("Groq API Key is required. Get one free at console.groq.com/keys")
                return None
            
            if not groq_key.strip().startswith("gsk_"):
                st.error("Invalid Groq API Key format. It should start with 'gsk_'")
                return None
            
            # Attempt signup
            user, error = auth_service.signup(
                name=name,
                email=email,
                password=password,
                groq_api_key=groq_key.strip()
            )
            
            if user:
                st.success("Account created! Logging in...")
                st.session_state.user_id = user.id
                st.session_state.user_name = user.name
                st.session_state.user_email = user.email
                st.session_state.groq_api_key = user.groq_api_key
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error(error)
    
    return None


def logout():
    """Log out the current user."""
    keys_to_remove = [
        "user_id", "user_name", "user_email", "groq_api_key",
        "authenticated", "current_chat_id", "messages", "auth_mode"
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    st.rerun()


def get_current_user() -> User | None:
    """Get the currently authenticated user."""
    if not st.session_state.get("authenticated", False):
        return None
    
    user_id = st.session_state.get("user_id")
    if not user_id:
        return None
    
    return User(
        id=user_id,
        name=st.session_state.get("user_name", ""),
        email=st.session_state.get("user_email", ""),
        groq_api_key=st.session_state.get("groq_api_key")
    )


def require_auth():
    """Check if user is authenticated."""
    return st.session_state.get("authenticated", False)
