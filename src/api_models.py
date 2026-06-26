"""API inference layer for Phase 2 (capability-matched cross-family judge).

Drop-in replacement for the local `models.LM` interface so the existing
pipeline (phaselib.solve_record, prompts.*, error_locator.*) runs unchanged:

    lm = APILM("gpt-4o-mini")
    text = lm.chat(messages)                 # greedy completion
    ppl  = lm.perplexity(ctx, response)      # teacher-forced ppl, where available

Two providers, one common interface:
  - OpenAI   (Chat Completions; logprobs on *generated* tokens only)
  - Anthropic (Messages API; no token logprobs)

Design constraints carried over from the project:
  - **Mandatory disk cache** keyed by a content hash of the full request
    (provider, model, messages, decoding params). Identical requests are never
    re-billed; a crashed run resumes for free by re-issuing the same requests
    and hitting cache. Optional (role, problem_id) metadata is stored alongside
    for transparency but is NOT part of the key — the key is the exact request,
    which is strictly safer for reproducibility.
  - **Determinism**: temperature=0 everywhere (greedy), so cache reuse is sound.
  - **Robustness**: bounded exponential backoff on rate-limit / transient errors.

Teacher-forced perplexity note (read once): neither gpt-4o-class chat models nor
Claude expose logprobs for an *arbitrary provided* assistant string (no "echo"),
so the §4 familiarity proxy used in the local run is **not computable on these
chat APIs**. `perplexity()` returns NaN unless a model genuinely supports it.
This is reported honestly in the writeup; the decisive Phase-2 quantity (excess
co-location under capability matching) does not depend on it. See models.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

from dotenv import load_dotenv

# Load .env from the project root (keys live there, gitignored).
_ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(_ROOT, ".env"))

CACHE_DIR = os.path.join(_ROOT, "results", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model registry. Three roles for Phase 2:
#   answerer M  = same-family SMALL  (OpenAI gpt-4o-mini)
#   judge same  = same-family LARGE  (OpenAI gpt-4o)        <- shared origin w/ M
#   judge cross = different provider (Anthropic Claude)     <- capability-matched
# Exact IDs are pinned in models.md after the Step-0 probe; calibration (Step 1)
# may swap the cross tier (haiku<->sonnet) to match gpt-4o's accuracy.
# ---------------------------------------------------------------------------
API_MODELS = {
    # OpenAI
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    "gpt-4o": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
    "gpt-4.1-mini": {"provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
    "gpt-4.1": {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
    # Anthropic (cross-family judge candidates; IDs confirmed available to this
    # account 2026-06-26 via models.list — the 3.5 line is retired).
    "claude-haiku": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "claude-sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
}


def resolve(key_or_spec):
    """Accept a registry key or an explicit {provider, model} dict."""
    if isinstance(key_or_spec, dict):
        return key_or_spec
    if key_or_spec in API_MODELS:
        return API_MODELS[key_or_spec]
    raise KeyError(f"unknown model key {key_or_spec!r}; add it to API_MODELS")


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".json")


class APILM:
    """One API model behind the local-LM interface, with mandatory caching."""

    def __init__(self, key_or_spec, max_retries: int = 6):
        spec = resolve(key_or_spec)
        self.key = key_or_spec if isinstance(key_or_spec, str) else spec["model"]
        self.provider = spec["provider"]
        self.model = spec["model"]
        self.max_retries = max_retries
        self.device = self.provider  # for log-compat with local LM.device
        self._client = None  # lazy

    # -- client (lazy so importing the module never needs a key) ----------
    @property
    def client(self):
        if self._client is None:
            # Explicit per-request timeout so a single stalled call can't hang the
            # whole run; the SDK default (up to 600s) is far too long. Our own
            # backoff retries on timeout. 90s comfortably covers a long greedy
            # 16-step CoT; anything longer is a stall, not real work.
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(timeout=90.0, max_retries=0)
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(timeout=90.0, max_retries=0)
            else:
                raise ValueError(f"unknown provider {self.provider}")
        return self._client

    # -- generation -------------------------------------------------------
    def chat(self, messages: list[dict], max_new_tokens: int = 1200,
             role: str | None = None, problem_id=None) -> str:
        """Greedy (temperature 0) completion, cached by request content.

        `role`/`problem_id` are recorded in the cache file for traceability but
        are NOT part of the cache key (the request content is)."""
        payload = {
            "provider": self.provider, "model": self.model,
            "messages": messages, "max_tokens": max_new_tokens,
            "temperature": 0, "op": "chat",
        }
        key = _cache_key(payload)
        path = _cache_path(key)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["text"]

        text, meta = self._chat_uncached(messages, max_new_tokens)
        record = {"text": text, "meta": meta, "role": role,
                  "problem_id": problem_id, "key": key,
                  "provider": self.provider, "model": self.model}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f)
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves junk
        return text

    def _chat_uncached(self, messages, max_new_tokens):
        if self.provider == "openai":
            return self._openai_chat(messages, max_new_tokens)
        return self._anthropic_chat(messages, max_new_tokens)

    def _with_retries(self, fn):
        delay = 2.0
        last = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - provider-specific; backoff on all
                last = e
                name = type(e).__name__
                transient = any(s in name for s in
                                ("RateLimit", "APIConnection", "APITimeout",
                                 "InternalServer", "Overloaded", "APIStatus"))
                if not transient or attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise last  # unreachable

    # -- OpenAI -----------------------------------------------------------
    def _openai_chat(self, messages, max_new_tokens):
        def call():
            return self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=0, max_tokens=max_new_tokens, seed=0,
            )
        resp = self._with_retries(call)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        meta = {"finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None)}
        return (choice.message.content or "").strip(), meta

    # -- Anthropic --------------------------------------------------------
    def _anthropic_chat(self, messages, max_new_tokens):
        # Anthropic takes the system prompt as a top-level arg, not a message.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m["role"] != "system"]

        def call():
            kwargs = dict(model=self.model, max_tokens=max_new_tokens,
                          temperature=0, messages=convo)
            if system:
                kwargs["system"] = system
            return self.client.messages.create(**kwargs)
        resp = self._with_retries(call)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        meta = {"stop_reason": resp.stop_reason,
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None)}
        return text.strip(), meta

    # -- perplexity (familiarity proxy) -----------------------------------
    def perplexity(self, context_messages: list[dict], response_text: str) -> float:
        """Teacher-forced perplexity of `response_text` given the context.

        Not computable on gpt-4o-class chat models or Claude (no echo /
        teacher-forced logprobs for a provided assistant string). Returns NaN,
        which the metrics layer already treats as "no perplexity available".
        Kept as a hook so an echo-capable model could slot in later."""
        return float("nan")

    def close(self):  # interface-compat with local LM; nothing to free
        pass


# -- cache accounting (used by the cost gate / dry run) --------------------
def cache_cost_summary() -> dict:
    """Aggregate token usage across all cached calls, for spend extrapolation."""
    per_model: dict = {}
    n_files = 0
    for fn in os.listdir(CACHE_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(CACHE_DIR, fn)) as f:
                rec = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        n_files += 1
        m = rec.get("model", "?")
        meta = rec.get("meta") or {}
        d = per_model.setdefault(m, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        d["calls"] += 1
        d["prompt_tokens"] += meta.get("prompt_tokens") or 0
        d["completion_tokens"] += meta.get("completion_tokens") or 0
    return {"n_cached_calls": n_files, "per_model": per_model}
