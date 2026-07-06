from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol


class ModelProvider(Protocol):
    name: str

    def json_tool(
        self,
        prompt: str,
        temperature: float = 0.2,
        timeout: int = 18,
    ) -> tuple[dict[str, Any] | None, str]:
        ...

    def final_comment(
        self,
        prompt: str,
        temperature: float = 0.4,
        timeout: int = 12,
    ) -> tuple[str | None, str]:
        ...


def extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        import re

        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    import re

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_gemini_text(data: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if part.get("text"):
                text_parts.append(str(part["text"]))
    return "\n".join(text_parts).strip()


def extract_openai_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"]).strip()

    text_parts: list[str] = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                text_parts.append(str(content["text"]))
    return "\n".join(text_parts).strip()


class GeminiProvider:
    name = "Gemini"

    def _request(self, prompt: str, temperature: float, timeout: int, json_mode: bool) -> tuple[str | None, str]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None, "GEMINI_API_KEY 없음"

        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        generation_config: dict[str, Any] = {"temperature": temperature}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model, safe='-_.')}:generateContent"
            f"?key={urllib.parse.quote(api_key)}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            return None, str(error)
        return extract_gemini_text(data), "Gemini LLM"

    def json_tool(self, prompt: str, temperature: float = 0.2, timeout: int = 18) -> tuple[dict[str, Any] | None, str]:
        text, mode = self._request(prompt, temperature, timeout, json_mode=True)
        if not text:
            return None, mode
        parsed = extract_json_payload(text)
        if parsed is None:
            return None, "LLM JSON 파싱 실패"
        return parsed, mode

    def final_comment(self, prompt: str, temperature: float = 0.4, timeout: int = 12) -> tuple[str | None, str]:
        return self._request(prompt, temperature, timeout, json_mode=False)


class OpenAIProvider:
    name = "OpenAI"

    def _request(self, prompt: str, temperature: float, timeout: int) -> tuple[str | None, str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, "OPENAI_API_KEY 없음"

        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "input": prompt,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            return None, str(error)
        return extract_openai_text(data), "OpenAI LLM"

    def json_tool(self, prompt: str, temperature: float = 0.2, timeout: int = 18) -> tuple[dict[str, Any] | None, str]:
        text, mode = self._request(prompt, temperature, timeout)
        if not text:
            return None, mode
        parsed = extract_json_payload(text)
        if parsed is None:
            return None, "LLM JSON 파싱 실패"
        return parsed, mode

    def final_comment(self, prompt: str, temperature: float = 0.4, timeout: int = 12) -> tuple[str | None, str]:
        return self._request(prompt, temperature, timeout)


def get_model_provider(name: str | None = None) -> ModelProvider:
    provider_name = (name or os.getenv("MODEL_PROVIDER") or "gemini").strip().lower()
    if provider_name in {"openai", "open-ai"}:
        return OpenAIProvider()
    return GeminiProvider()

