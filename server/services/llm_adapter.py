"""LLM adapter interface and provider implementations for PoC."""

from typing import List, Dict, Any, Optional
import os
import json
import logging
import time
import requests

logger = logging.getLogger("llm_adapter")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5


def _retry_request(method, *args, **kwargs) -> requests.Response:
    """Execute an HTTP request with exponential backoff on retryable status codes."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = method(*args, **kwargs)
            if resp.status_code not in _RETRYABLE_STATUS or attempt == _MAX_RETRIES - 1:
                resp.raise_for_status()
                return resp
            logger.warning("Retryable HTTP %d on attempt %d", resp.status_code, attempt + 1)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt == _MAX_RETRIES - 1:
                raise
            logger.warning("Request error on attempt %d: %s", attempt + 1, e)
        time.sleep(_BACKOFF_BASE * (2 ** attempt))
    raise last_exc or RuntimeError("Retry exhausted")


def _extract_usage(data: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage from API response if available."""
    usage = data.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


class LLMAdapter:
    """Interface for LLM providers."""

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2,
                 system: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError()

    def batch_generate(self, prompts: List[str], system: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def moderate(self, text: str) -> Dict[str, Any]:
        """Optional moderation call. Returns moderation result dict."""
        return {"flagged": False, "reason": None}


class MockLLMAdapter(LLMAdapter):
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2,
                 system: Optional[str] = None) -> Dict[str, Any]:
        return {"text": "MOCK_ANSWER: This is a mock response.", "tokens": 5, "usage": {"prompt_tokens": 0, "completion_tokens": 5, "total_tokens": 5}}

    def batch_generate(self, prompts: List[str], system: Optional[str] = None) -> List[Dict[str, Any]]:
        return [self.generate(p, system=system) for p in prompts]


class AzureLLMAdapter(LLMAdapter):
    def __init__(self, endpoint: str, key: str, deployment: str, embedding_endpoint: str | None, embedding_key: str | None, embedding_deployment: str | None):
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.deployment = deployment
        self.embedding_endpoint = (embedding_endpoint or endpoint).rstrip("/")
        self.embedding_key = embedding_key or key
        self.embedding_deployment = embedding_deployment

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2,
                 system: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-01"
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        resp = _retry_request(requests.post, url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        choices = data.get("choices", [])
        text = ""
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {})
            text = msg.get("content", "") if isinstance(msg, dict) else ""
        usage = _extract_usage(data)
        return {"text": text or "", "raw": data, "usage": usage}

    def batch_generate(self, prompts: List[str], system: Optional[str] = None) -> List[Dict[str, Any]]:
        return [self.generate(p, system=system) for p in prompts]

    def embed(self, text: str) -> List[float]:
        if not self.embedding_deployment:
            raise RuntimeError("EmbeddingDeploymentName not configured")
        url = f"{self.embedding_endpoint}/openai/deployments/{self.embedding_deployment}/embeddings?api-version=2024-02-01"
        headers = {"api-key": self.embedding_key, "Content-Type": "application/json"}
        payload = {"input": text}
        resp = _retry_request(requests.post, url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        embedding = data.get("data", [{}])[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Azure embedding response did not contain a valid embedding vector")
        return embedding


class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible chat APIs (including Groq-style endpoints)."""

    def __init__(self, endpoint: str, key: str, model: str, embedding_model: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.model = model
        self.embedding_model = embedding_model

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def _stream_enabled(self) -> bool:
        return (os.getenv("LLM_STREAM") or os.getenv("OPENAI_STREAM") or "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _extract_text(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return ""

        choice0 = choices[0] or {}
        message = choice0.get("message") or {}

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
            if parts:
                return " ".join(parts)

        # Some reasoning-style models provide useful text here.
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        text = choice0.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        return ""

    def _extract_stream_delta_text(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return ""

        choice0 = choices[0] or {}
        delta = choice0.get("delta") or {}

        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
            if parts:
                return "".join(parts)

        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            return reasoning

        return ""

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2,
                 system: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.endpoint}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self._stream_enabled():
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            resp = _retry_request(requests.post, url, headers=self._headers(), json=stream_payload, timeout=60, stream=True)

            chunks: List[str] = []
            last_obj: Dict[str, Any] = {}
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not isinstance(line, str):
                    try:
                        line = line.decode("utf-8")
                    except Exception:
                        continue
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                last_obj = obj
                delta_text = self._extract_stream_delta_text(obj)
                if delta_text:
                    chunks.append(delta_text)

            joined = "".join(chunks).strip()
            usage = _extract_usage(last_obj)
            if joined:
                return {"text": joined, "raw": last_obj, "usage": usage}

            # fallback in case stream provided no deltas
            return {"text": self._extract_text(last_obj), "raw": last_obj, "usage": usage}

        resp = _retry_request(requests.post, url, headers=self._headers(), json=payload, timeout=30)
        data = resp.json()
        content = self._extract_text(data)
        usage = _extract_usage(data)
        return {"text": content or "", "raw": data, "usage": usage}

    def batch_generate(self, prompts: List[str], system: Optional[str] = None) -> List[Dict[str, Any]]:
        return [self.generate(p, system=system) for p in prompts]

    def moderate(self, text: str) -> Dict[str, Any]:
        # Not all OpenAI-compatible providers expose moderation.
        try:
            url = f"{self.endpoint}/moderations"
            payload = {"model": "omni-moderation-latest", "input": text}
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=20)
            if resp.status_code >= 400:
                return {"flagged": False, "reason": "moderation_not_available"}
            data = resp.json()
            result = data.get("results", [{}])[0] if isinstance(data.get("results"), list) else {}
            return {"flagged": bool(result.get("flagged")), "reason": None}
        except Exception:
            return {"flagged": False, "reason": "moderation_not_available"}

    def embed(self, text: str) -> List[float]:
        if not self.embedding_model:
            raise RuntimeError("Embedding model not configured for OpenAI-compatible provider")
        url = f"{self.endpoint}/embeddings"
        payload = {"model": self.embedding_model, "input": text}
        resp = _retry_request(requests.post, url, headers=self._headers(), json=payload, timeout=30)
        data = resp.json()
        embedding = data.get("data", [{}])[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("OpenAI embedding response did not contain a valid embedding vector")
        return embedding


def get_llm_adapter() -> LLMAdapter:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()

    if provider == "mock":
        return MockLLMAdapter()

    endpoint = os.getenv("OpenAIEndpoint")
    key = os.getenv("OpenAIKey")
    # Support both corrected and legacy (typo) env var names
    deployment_or_model = os.getenv("OpenAIDeploymentName") or os.getenv("OpenAIDeplymentName")
    embedding_endpoint = os.getenv("EmbeddingEndpoint")
    embedding_key = os.getenv("EmbeddingKey")
    embedding_deployment = os.getenv("EmbeddingDeploymentName")

    if provider in ("azure", "openai-azure"):
        if not endpoint or not key or not deployment_or_model:
            raise RuntimeError("Azure OpenAI settings not configured in env")
        return AzureLLMAdapter(
            endpoint=endpoint,
            key=key,
            deployment=deployment_or_model,
            embedding_endpoint=embedding_endpoint,
            embedding_key=embedding_key,
            embedding_deployment=embedding_deployment,
        )

    if provider in ("openai-compatible", "openai", "groq"):
        if not endpoint or not key or not deployment_or_model:
            raise RuntimeError("OpenAI-compatible settings not configured in env")
        return OpenAICompatibleAdapter(
            endpoint=endpoint,
            key=key,
            model=deployment_or_model,
            embedding_model=embedding_deployment,
        )

    raise RuntimeError(f"LLM provider '{provider}' not implemented")
