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
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert assistant that provides direct, informative answers based ONLY on the provided document context.

CRITICAL RULES:
1. ANSWER DIRECTLY - Don't say "according to the document" or "the context mentions". Just provide the answer.
2. BE INFORMATIVE - Give complete, detailed answers with specific facts, numbers, steps, or explanations from the context.
3. STRUCTURE WELL - Use bullet points, numbered lists, or paragraphs as appropriate for clarity.
4. STAY FACTUAL - Only include information that exists in the provided context. Never make up information.

WHEN INFORMATION IS NOT FOUND:
If the context does NOT contain information to answer the question, respond with a friendly message like:
"I couldn't find information about [topic] in your uploaded documents. The documents I have access to don't seem to cover this topic. You might want to:
• Upload additional documents that contain this information
• Rephrase your question
• Ask about a different aspect of the topic"

IMPORTANT: Do NOT hallucinate or guess. If it's not in the context, say so politely."""

    RAG_PROMPT_TEMPLATE = """You have access to the following document excerpts. Use ONLY this information to answer the question.

=== DOCUMENT CONTEXT ===
{context}
========================

PREVIOUS CONVERSATION:
{history}

USER'S QUESTION: {question}

INSTRUCTIONS:
1. If the answer IS in the context: Provide a direct, complete answer. Include specific details, facts, numbers, or steps. Don't reference "the document" - just answer naturally as if you know this information.

2. If the answer is NOT in the context: Politely inform the user that this information isn't available in their uploaded documents. Suggest they might upload relevant documents or rephrase their question.

3. If the answer is PARTIALLY in the context: Provide what you can find, then clearly state what additional information would be needed.

Your response:"""

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the LLM service.
        
        Args:
            provider: Provider to use ("groq" or "ollama")
            api_key: API key for the provider (required for Groq)
        """
        self.provider_name = provider or config.LLM_PROVIDER
        self.api_key = api_key  # Store user's API key
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
            # Use the user's API key passed during initialization
            return GroqProvider(api_key=self.api_key)
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
    
    def generate_similar_queries(self, query: str, num_queries: int = 5) -> list[str]:
        """
        Generate similar/related queries for multi-query retrieval.
        
        This helps improve retrieval by searching with multiple perspectives
        of the same question.
        
        Args:
            query: Original user query
            num_queries: Number of similar queries to generate
            
        Returns:
            List of similar queries (including the original)
        """
        prompt = f"""Given the following user question, generate {num_queries - 1} alternative versions of this question that could help retrieve relevant information from a document database.

The alternative questions should:
1. Rephrase the original question differently
2. Break down complex questions into simpler parts
3. Use synonyms or related terms
4. Approach the question from different angles

Original question: "{query}"

Return ONLY the alternative questions, one per line, without numbering or bullets. Do not include the original question."""

        try:
            response = self.provider.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=300
            )
            
            # Parse the response - split by newlines and clean up
            similar_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove common prefixes like "1.", "-", "*", etc.
                line = line.lstrip('0123456789.-)*• ')
                if line and len(line) > 5:  # Filter out very short lines
                    similar_queries.append(line)
            
            # Limit to requested number and always include original first
            similar_queries = similar_queries[:num_queries - 1]
            
            # Return original query first, then the generated ones
            return [query] + similar_queries
            
        except Exception as e:
            # On error, just return the original query
            print(f"Error generating similar queries: {e}")
            return [query]
    
    # Model context limits (in tokens) - conservative estimates
    # Keeping buffer for system prompt, user question, and response
    MODEL_CONTEXT_LIMITS = {
        "llama-3.1-8b-instant": 120000,      # 128K context, leaving buffer
        "llama-3.1-70b-versatile": 120000,   # 128K context
        "mixtral-8x7b-32768": 28000,         # 32K context
        "llama3.1": 120000,                   # Ollama llama3.1
        "mistral": 28000,                     # Ollama mistral
        "default": 8000                       # Safe fallback
    }
    
    # Characters per token (rough estimate)
    CHARS_PER_TOKEN = 4
    
    @staticmethod
    def format_context(search_results, max_context_chars: int = None) -> str:
        """
        Format search results into context string with optional length limit.
        
        Args:
            search_results: List of SearchResult objects
            max_context_chars: Maximum characters for context (optional)
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "[NO RELEVANT DOCUMENTS FOUND - The search did not find any documents matching the user's query. Please inform the user that no relevant information was found in their uploaded documents.]"
        
        # Default max context: ~20K tokens worth of chars (80K chars) - very safe
        if max_context_chars is None:
            max_context_chars = 80000
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results, 1):
            source_info = f"[Document {i}: {result.document.source_file}"
            if result.document.page_number:
                source_info += f", Page {result.document.page_number}"
            source_info += "]"
            
            chunk_content = f"{source_info}\n{result.document.content}"
            chunk_length = len(chunk_content) + 7  # +7 for separator "\n\n---\n\n"
            
            # Check if adding this chunk would exceed limit
            if current_length + chunk_length > max_context_chars:
                # Try to fit partial content
                remaining_space = max_context_chars - current_length - 50  # buffer
                if remaining_space > 500:  # Only add if meaningful content fits
                    truncated = f"{source_info}\n{result.document.content[:remaining_space]}... [truncated]"
                    context_parts.append(truncated)
                break
            
            context_parts.append(chunk_content)
            current_length += chunk_length
        
        return "\n\n---\n\n".join(context_parts)

