"""Authentication service for user login and signup."""

import hashlib
import re
from typing import Optional, Tuple
from dataclasses import dataclass

from services.database import db


@dataclass
class User:
    """User data class."""
    id: int
    name: str
    email: str
    groq_api_key: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create User from database row dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            email=data["email"],
            groq_api_key=data.get("groq_api_key")
        )


class AuthService:
    """Service for handling user authentication."""
    
    @staticmethod
    def normalize_email(email: str) -> str:
        """
        Normalize email address.
        - Convert to lowercase
        - Remove + aliases (e.g., user+alias@gmail.com -> user@gmail.com)
        - Strip whitespace
        
        Args:
            email: Raw email address
            
        Returns:
            Normalized email address
        """
        email = email.strip().lower()
        
        # Split email into local part and domain
        if "@" not in email:
            return email
        
        local_part, domain = email.rsplit("@", 1)
        
        # Remove + alias from local part
        if "+" in local_part:
            local_part = local_part.split("+")[0]
        
        return f"{local_part}@{domain}"
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic email regex pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not email:
            return False, "Email is required"
        
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not password:
            return False, "Password is required"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        return True, ""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using SHA-256 with salt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        # Using a simple salt - in production, use bcrypt or argon2
        salt = "info_finder_salt_2024"
        salted_password = f"{salt}{password}{salt}"
        return hashlib.sha256(salted_password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            password_hash: Stored hash
            
        Returns:
            True if password matches
        """
        return AuthService.hash_password(password) == password_hash
    
    @staticmethod
    def validate_groq_key(groq_key: str) -> Tuple[bool, str]:
        """
        Validate Groq API key format.
        
        Args:
            groq_key: Groq API key to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not groq_key or not groq_key.strip():
            return False, "Groq API Key is required"
        
        groq_key = groq_key.strip()
        
        if not groq_key.startswith("gsk_"):
            return False, "Invalid Groq API Key format. It should start with 'gsk_'"
        
        if len(groq_key) < 20:
            return False, "Groq API Key appears to be too short"
        
        return True, ""
    
    def signup(
        self,
        name: str,
        email: str,
        password: str,
        groq_api_key: str
    ) -> Tuple[Optional[User], str]:
        """
        Register a new user.
        
        Args:
            name: User's name
            email: User's email
            password: User's password
            groq_api_key: Required Groq API key
            
        Returns:
            Tuple of (User object or None, error message)
        """
        # Validate name
        if not name or len(name.strip()) < 2:
            return None, "Name must be at least 2 characters"
        
        name = name.strip()
        
        # Validate and normalize email
        is_valid, error = self.validate_email(email)
        if not is_valid:
            return None, error
        
        normalized_email = self.normalize_email(email)
        
        # Check if email already exists
        if db.email_exists(normalized_email):
            return None, "An account with this email already exists"
        
        # Validate password
        is_valid, error = self.validate_password(password)
        if not is_valid:
            return None, error
        
        # Validate Groq API key (required)
        is_valid, error = self.validate_groq_key(groq_api_key)
        if not is_valid:
            return None, error
        
        groq_api_key = groq_api_key.strip()
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Create user
        user_id = db.create_user(
            name=name,
            email=normalized_email,
            password_hash=password_hash,
            groq_api_key=groq_api_key
        )
        
        if user_id is None:
            return None, "Failed to create account. Please try again."
        
        # Return user object
        return User(
            id=user_id,
            name=name,
            email=normalized_email,
            groq_api_key=groq_api_key
        ), ""
    
    def login(self, email: str, password: str) -> Tuple[Optional[User], str]:
        """
        Authenticate user login.
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Tuple of (User object or None, error message)
        """
        # Validate inputs
        if not email or not password:
            return None, "Email and password are required"
        
        # Normalize email
        normalized_email = self.normalize_email(email)
        
        # Get user from database
        user_data = db.get_user_by_email(normalized_email)
        
        if user_data is None:
            return None, "Invalid email or password"
        
        # Verify password
        if not self.verify_password(password, user_data["password_hash"]):
            return None, "Invalid email or password"
        
        # Return user object
        return User.from_dict(user_data), ""
    
    def get_user(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None
        """
        user_data = db.get_user_by_id(user_id)
        if user_data:
            return User.from_dict(user_data)
        return None
    
    def update_groq_key(self, user_id: int, groq_api_key: str) -> Tuple[bool, str]:
        """
        Update user's Groq API key.
        
        Args:
            user_id: User ID
            groq_api_key: New Groq API key
            
        Returns:
            Tuple of (success, error_message)
        """
        # Validate the new key
        is_valid, error = self.validate_groq_key(groq_api_key)
        if not is_valid:
            return False, error
        
        groq_api_key = groq_api_key.strip()
        success = db.update_user_groq_key(user_id, groq_api_key)
        
        if success:
            return True, ""
        return False, "Failed to update API key"


# Global auth service instance
auth_service = AuthService()

