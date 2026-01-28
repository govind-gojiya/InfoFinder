"""Data models and schemas for the application."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import uuid


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class Document:
    """Represents a chunked document piece."""
    id: str
    content: str
    content_type: ContentType
    metadata: dict = field(default_factory=dict)
    source_file: str = ""
    page_number: int = 0
    chunk_index: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    document: Document
    score: float
    search_type: str = "hybrid"  # "vector", "keyword", or "hybrid"
    
    def to_dict(self) -> dict:
        return {
            "document_id": self.document.id,
            "content": self.document.content,
            "score": self.score,
            "search_type": self.search_type,
            "metadata": self.document.metadata,
            "source_file": self.document.source_file,
            "page_number": self.document.page_number
        }


@dataclass
class Message:
    """Represents a chat message."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: list[SearchResult] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": [s.to_dict() for s in self.sources] if self.sources else []
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            sources=[]  # Sources are not restored from dict
        )


@dataclass
class Chat:
    """Represents a chat session with history and documents."""
    id: str
    title: str
    user_id: int = 0  # Owner user ID
    messages: list[Message] = field(default_factory=list)
    document_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_message(self, role: MessageRole, content: str, sources: list[SearchResult] = None) -> Message:
        """Add a message to the chat."""
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            sources=sources or []
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_conversation_history(self, max_messages: int = 10) -> list[dict]:
        """Get recent conversation history for context."""
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": m.role.value, "content": m.content} for m in recent_messages]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "document_count": self.document_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Chat":
        chat = cls(
            id=data["id"],
            title=data["title"],
            user_id=data.get("user_id", 0),
            document_count=data.get("document_count", 0),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        )
        chat.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return chat

