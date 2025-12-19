from __future__ import annotations

import os
from typing import List

from pydantic import BaseModel

from vihermes.Agents.models import AnswerResponse
from vihermes.lawrag.models import RetrievalResult
from vihermes.Agents.prompt import build_prompt


class LLMClient(BaseModel):
    model: str = "gpt-4o"
    api_key: str | None = None
    provider: str = "openai"  # "openai" or "gemini"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, provider: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        
        # Auto-detect provider from model name
        if provider is None:
            if "gemini" in model.lower():
                self.provider = "gemini"
                self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            else:
                self.provider = "openai"
                self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            self.provider = provider
            if provider == "gemini":
                self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            else:
                self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def complete(self, prompt: str, temperature: float | None = None) -> str:
        """
        Complete prompt using OpenAI or Gemini API.
        
        Args:
            prompt: The prompt to complete
            temperature: Temperature for generation (None = use default: 0.3)
        """
        if not self.api_key:
            if self.provider == "gemini":
                raise ValueError("Gemini API key is required. Set GEMINI_API_KEY env var or pass api_key parameter.")
            else:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key parameter.")

        if self.provider == "gemini":
            return self._complete_gemini(prompt, temperature)
        else:
            return self._complete_openai(prompt, temperature)

    def _complete_openai(self, prompt: str, temperature: float | None = None) -> str:
        """Complete prompt using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        client = OpenAI(api_key=self.api_key)
        temp = temperature if temperature is not None else 0.3
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia pháp lý Việt Nam."},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
        )
        return response.choices[0].message.content or ""

    def _complete_gemini(self, prompt: str, temperature: float | None = None) -> str:
        """Complete prompt using Gemini API."""
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")

        genai.configure(api_key=self.api_key)  # type: ignore
        
        # Try to create model, fallback to gemini-1.5-pro if model not available
        try:
            model = genai.GenerativeModel(self.model)  # type: ignore
        except Exception:
            # Fallback to gemini-1.5-pro if specified model is not available
            if "2.0" in self.model or "2.5" in self.model:
                print(f"⚠️  Model {self.model} not available, falling back to gemini-1.5-pro")
                model = genai.GenerativeModel("gemini-1.5-pro")  # type: ignore
            else:
                raise
        
        # Combine system message and prompt
        full_prompt = f"You are a Vietnamese legal expert.\n\n{prompt}"
        
        temp = temperature if temperature is not None else 0.3
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(  # type: ignore
                temperature=temp,
            )
        )
        
        return response.text if response.text else ""


class GraphRAGEngine:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(self, query: str, retrieved: List[RetrievalResult]) -> AnswerResponse:
        prompt = build_prompt(query=query, results=retrieved)
        answer_text = self._llm.complete(prompt)

        sources = [
            {"type": "Article", "id": r.chunk.document_id, "chunk": r.chunk.id}
            for r in retrieved
        ]
        trace = ["Hybrid retrieval: vector search + 1-hop graph expansion"]
        return AnswerResponse(answer=answer_text, sources=sources, graph_trace=trace)

