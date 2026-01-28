"""LLM service supporting Groq and Ollama."""

from typing import Generator, Optional
from abc import ABC, abstractmethod

import config


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider (free tier available)."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key
            model: Model name to use
        """
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get a free key at https://console.groq.com/"
            )
        
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate a response using Groq."""
        messages = self._build_messages(prompt, system_prompt, conversation_history)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """Generate a streaming response using Groq."""
        messages = self._build_messages(prompt, system_prompt, conversation_history)
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None
    ) -> list[dict]:
        """Build messages list for the API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        return messages


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider (local, free)."""
    
    def __init__(self, base_url: str = None, model: str = None):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.OLLAMA_MODEL
        
        import ollama
        self.client = ollama.Client(host=self.base_url)
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate a response using Ollama."""
        messages = self._build_messages(prompt, system_prompt, conversation_history)
        
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        
        return response["message"]["content"]
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """Generate a streaming response using Ollama."""
        messages = self._build_messages(prompt, system_prompt, conversation_history)
        
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            },
            stream=True
        )
        
        for chunk in stream:
            if chunk["message"]["content"]:
                yield chunk["message"]["content"]
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_history: list[dict] = None
    ) -> list[dict]:
        """Build messages list for Ollama."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        return messages


class LLMService:
    """
    Main LLM service that manages providers.
    Supports Groq (free cloud) and Ollama (free local).
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context. 
Your responses should be:
1. Accurate and based on the given context
2. Clear and well-structured
3. Honest about uncertainty - if the context doesn't contain enough information, say so

When answering:
- Reference specific parts of the context when relevant
- If multiple sources provide different information, acknowledge this
- Be concise but thorough"""

    RAG_PROMPT_TEMPLATE = """Based on the following context, please answer the user's question.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Please provide a helpful and accurate response based on the context provided. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information you can."""

    def __init__(self, provider: str = None):
        """
        Initialize the LLM service.
        
        Args:
            provider: Provider to use ("groq" or "ollama")
        """
        self.provider_name = provider or config.LLM_PROVIDER
        self._provider: Optional[BaseLLMProvider] = None
    
    @property
    def provider(self) -> BaseLLMProvider:
        """Get or initialize the LLM provider."""
        if self._provider is None:
            self._provider = self._init_provider()
        return self._provider
    
    def _init_provider(self) -> BaseLLMProvider:
        """Initialize the appropriate LLM provider."""
        if self.provider_name == "groq":
            return GroqProvider()
        elif self.provider_name == "ollama":
            return OllamaProvider()
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")
    
    def generate_response(
        self,
        question: str,
        context: str = "",
        conversation_history: list[dict] = None,
        system_prompt: str = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response for a RAG query.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            conversation_history: Previous conversation messages
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Format conversation history
        history_str = ""
        if conversation_history:
            history_str = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history[-6:]  # Last 6 messages
            ])
        
        # Build the prompt
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context if context else "No context provided.",
            history=history_str if history_str else "No previous conversation.",
            question=question
        )
        
        return self.provider.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature
        )
    
    def generate_response_stream(
        self,
        question: str,
        context: str = "",
        conversation_history: list[dict] = None,
        system_prompt: str = None,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response for a RAG query.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            conversation_history: Previous conversation messages
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature
            
        Yields:
            Response tokens
        """
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Format conversation history
        history_str = ""
        if conversation_history:
            history_str = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history[-6:]
            ])
        
        # Build the prompt
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context if context else "No context provided.",
            history=history_str if history_str else "No previous conversation.",
            question=question
        )
        
        yield from self.provider.generate_stream(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature
        )
    
    def generate_chat_title(self, first_message: str) -> str:
        """
        Generate a title for a chat based on the first message.
        
        Args:
            first_message: The first user message in the chat
            
        Returns:
            Generated title
        """
        prompt = f"""Generate a short, descriptive title (3-6 words) for a conversation that starts with this message:

"{first_message}"

Respond with ONLY the title, nothing else."""
        
        try:
            title = self.provider.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=20
            )
            # Clean up the title
            title = title.strip().strip('"\'').strip()
            return title[:50]  # Limit length
        except Exception:
            # Fallback to truncated message
            return first_message[:30] + "..." if len(first_message) > 30 else first_message
    
    @staticmethod
    def format_context(search_results) -> str:
        """
        Format search results into context string.
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source_info = f"[Source {i}: {result.document.source_file}"
            if result.document.page_number:
                source_info += f", Page {result.document.page_number}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{result.document.content}")
        
        return "\n\n---\n\n".join(context_parts)

