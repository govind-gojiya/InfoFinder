"""SQLite database module for user management."""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
import config


class Database:
    """SQLite database manager for user data."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize database connection and create tables."""
        if self._initialized:
            return
        
        self.db_path = config.DATA_DIR / "info_finder.db"
        self._create_tables()
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Get a database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create necessary database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    groq_api_key TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on email for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
            """)
            
            conn.commit()
    
    def create_user(
        self,
        name: str,
        email: str,
        password_hash: str,
        groq_api_key: str = None
    ) -> Optional[int]:
        """
        Create a new user.
        
        Args:
            name: User's name
            email: User's normalized email
            password_hash: Hashed password
            groq_api_key: Optional Groq API key
            
        Returns:
            User ID if created, None if email already exists
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO users (name, email, password_hash, groq_api_key)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, email, password_hash, groq_api_key)
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Email already exists
            return None
    
    def get_user_by_email(self, email: str) -> Optional[dict]:
        """
        Get user by email.
        
        Args:
            email: Normalized email address
            
        Returns:
            User dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def update_user_groq_key(self, user_id: int, groq_api_key: str) -> bool:
        """
        Update user's Groq API key.
        
        Args:
            user_id: User ID
            groq_api_key: New Groq API key
            
        Returns:
            True if updated, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users 
                SET groq_api_key = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (groq_api_key, user_id)
            )
            return cursor.rowcount > 0
    
    def update_user_password(self, user_id: int, password_hash: str) -> bool:
        """
        Update user's password.
        
        Args:
            user_id: User ID
            password_hash: New hashed password
            
        Returns:
            True if updated, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users 
                SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (password_hash, user_id)
            )
            return cursor.rowcount > 0
    
    def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM users WHERE email = ?",
                (email,)
            )
            return cursor.fetchone() is not None


# Global database instance
db = Database()

