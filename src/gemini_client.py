"""Gemini API client with .env loading, response caching, retries, and usage tracking."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-flash-lite"


@dataclass
class GeminiResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False


def load_api_key() -> str:
    """Load GEMINI_API_KEY or GOOGLE_API_KEY from project .env or ~/.env without logging it."""
    project_env = Path(__file__).resolve().parents[1] / ".env"
    home_env = Path.home() / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=False)
    if home_env.exists():
        load_dotenv(home_env, override=False)

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Add it to ~/.env or project .env."
        )
    return key


class CachedGeminiClient:
    """Thread-safe Gemini API wrapper with JSONL disk caching and retry backoff."""

    def __init__(
        self,
        cache_path: Path,
        model: str = DEFAULT_MODEL,
        min_interval_s: float = 2.8,
        max_retries: int = 12,
        timeout_ms: int = 15000,
    ) -> None:
        self.api_key = load_api_key()
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=timeout_ms),
        )
        self.model = model
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._last_call_ts = 0.0
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._cache[rec["cache_key"]] = rec
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def make_key(
        model: str,
        system_instruction: str,
        prompt: str,
        json_mode: bool,
        disable_thinking: bool,
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "system": system_instruction,
                "prompt": prompt,
                "json_mode": json_mode,
                "disable_thinking": disable_thinking,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        json_mode: bool = False,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        disable_thinking: bool = True,
    ) -> GeminiResponse:
        cache_key = self.make_key(
            self.model, system_instruction, prompt, json_mode, disable_thinking
        )
        with self._lock:
            if cache_key in self._cache:
                rec = self._cache[cache_key]
                if rec.get("text"):
                    return GeminiResponse(
                        text=rec["text"],
                        input_tokens=int(rec.get("input_tokens", 0)),
                        output_tokens=int(rec.get("output_tokens", 0)),
                        latency_s=float(rec.get("latency_s", 0.0)),
                        cached=True,
                    )

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if disable_thinking:
            if "2.5" in self.model:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            elif "3." in self.model or "gemma-4" in self.model:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")

        config = types.GenerateContentConfig(**config_kwargs)

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            with self._lock:
                elapsed = time.monotonic() - self._last_call_ts
                if elapsed < self.min_interval_s:
                    time.sleep(self.min_interval_s - elapsed)
                self._last_call_ts = time.monotonic()

            t0 = time.monotonic()
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                dt = time.monotonic() - t0
                text = (resp.text or "").strip()
                if not text:
                    raise ValueError("Empty response text from model")
                usage = getattr(resp, "usage_metadata", None)
                in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
                out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)

                record = {
                    "cache_key": cache_key,
                    "model": self.model,
                    "text": text,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "latency_s": round(dt, 4),
                }
                with self._lock:
                    self._cache[cache_key] = record
                    with self.cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                return GeminiResponse(
                    text=text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_s=dt,
                    cached=False,
                )
            except Exception as exc:
                last_err = exc
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if "PerDay" in msg or "per day" in msg.lower():
                        raise RuntimeError(f"Daily quota exhausted for {self.model}: {msg[:240]}") from exc
                    m_delay = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                    sleep_s = float(m_delay.group(1)) + 1.0 if m_delay else min(45.0, 15.0 + 3.0 * attempt)
                else:
                    sleep_s = min(10.0, (1.4 ** attempt) + random.uniform(0.3, 1.2))
                if attempt >= 2:
                    print(f"[retry {attempt+1}/{self.max_retries} | {self.model}] {msg[:100]} -> sleeping {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)

        raise RuntimeError(f"Gemini API failed after {self.max_retries} attempts: {last_err}")
