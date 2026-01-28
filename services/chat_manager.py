"""Chat management service for persisting and loading chats."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from models.schemas import Chat, Message, MessageRole
import config


class ChatManager:
    """Service for managing chat sessions per user."""
    
    def __init__(self, user_id: int):
        """
        Initialize the chat manager for a specific user.
        
        Args:
            user_id: The ID of the current user
        """
        self.user_id = user_id
        self.chats_dir = config.CHATS_DIR
        self.chats_dir.mkdir(parents=True, exist_ok=True)
    
    def create_chat(self, title: str = "New Chat") -> Chat:
        """
        Create a new chat session for the current user.
        
        Args:
            title: Title for the new chat
            
        Returns:
            New Chat object
        """
        chat = Chat(
            id="",  # Will be auto-generated
            title=title,
            user_id=self.user_id
        )
        self.save_chat(chat)
        return chat
    
    def get_chat(self, chat_id: str) -> Optional[Chat]:
        """
        Get a chat by ID if it belongs to the current user.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Chat object or None if not found or not owned by user
        """
        chat_file = self.chats_dir / f"{chat_id}.json"
        if not chat_file.exists():
            return None
        
        try:
            with open(chat_file, "r") as f:
                data = json.load(f)
                chat = Chat.from_dict(data)
                
                # Security check: Only return if chat belongs to current user
                if chat.user_id != self.user_id:
                    return None
                
                return chat
        except Exception:
            return None
    
    def save_chat(self, chat: Chat):
        """
        Save a chat to disk.
        
        Args:
            chat: Chat object to save
        """
        # Ensure the chat belongs to the current user
        if chat.user_id != self.user_id:
            return
        
        chat_file = self.chats_dir / f"{chat.id}.json"
        with open(chat_file, "w") as f:
            json.dump(chat.to_dict(), f, indent=2)
    
    def delete_chat(self, chat_id: str):
        """
        Delete a chat and its associated data if owned by current user.
        
        Args:
            chat_id: Chat ID to delete
        """
        # First verify ownership
        chat = self.get_chat(chat_id)
        if not chat:
            return
        
        # Delete chat file
        chat_file = self.chats_dir / f"{chat_id}.json"
        if chat_file.exists():
            chat_file.unlink()
        
        # Delete vector store
        from services.vector_store import VectorStoreService
        VectorStoreService.delete_store(chat_id)
    
    def list_chats(self) -> list[Chat]:
        """
        List all chats for the current user, sorted by updated_at.
        
        Returns:
            List of Chat objects belonging to current user
        """
        chats = []
        for chat_file in self.chats_dir.glob("*.json"):
            try:
                with open(chat_file, "r") as f:
                    data = json.load(f)
                    chat = Chat.from_dict(data)
                    
                    # Only include chats belonging to current user
                    if chat.user_id == self.user_id:
                        chats.append(chat)
            except Exception:
                continue
        
        # Sort by updated_at (most recent first)
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        return chats
    
    def update_chat_title(self, chat_id: str, title: str):
        """
        Update the title of a chat if owned by current user.
        
        Args:
            chat_id: Chat ID
            title: New title
        """
        chat = self.get_chat(chat_id)
        if chat:
            chat.title = title
            chat.updated_at = datetime.now()
            self.save_chat(chat)
    
    def add_message(
        self,
        chat_id: str,
        role: MessageRole,
        content: str,
        sources: list = None
    ) -> Optional[Message]:
        """
        Add a message to a chat if owned by current user.
        
        Args:
            chat_id: Chat ID
            role: Message role
            content: Message content
            sources: Optional list of search results
            
        Returns:
            Added Message object or None
        """
        chat = self.get_chat(chat_id)
        if not chat:
            return None
        
        message = chat.add_message(role, content, sources)
        self.save_chat(chat)
        return message
    
    def update_document_count(self, chat_id: str, count: int):
        """
        Update the document count for a chat if owned by current user.
        
        Args:
            chat_id: Chat ID
            count: New document count
        """
        chat = self.get_chat(chat_id)
        if chat:
            chat.document_count = count
            self.save_chat(chat)
    
    def get_recent_chats(self, limit: int = 10) -> list[Chat]:
        """
        Get the most recent chats for current user.
        
        Args:
            limit: Maximum number of chats to return
            
        Returns:
            List of recent Chat objects
        """
        chats = self.list_chats()
        return chats[:limit]
    
    def search_chats(self, query: str) -> list[Chat]:
        """
        Search chats by title for current user.
        
        Args:
            query: Search query
            
        Returns:
            List of matching Chat objects
        """
        query = query.lower()
        chats = self.list_chats()
        return [c for c in chats if query in c.title.lower()]
    
    def verify_chat_ownership(self, chat_id: str) -> bool:
        """
        Verify if a chat belongs to the current user.
        
        Args:
            chat_id: Chat ID to verify
            
        Returns:
            True if chat belongs to current user
        """
        chat = self.get_chat(chat_id)
        return chat is not None
