"""Local Hugging Face inference wrapper (Option 1: open-weight, on-device).

One small instruct model at a time on Apple MPS (falls back to CPU). Provides:
  - chat(messages)           -> greedy, reproducible text completion
  - perplexity(ctx, text)    -> teacher-forced perplexity of `text` under this
                                model given `ctx` (the §4 familiarity proxy)

We load models one-at-a-time (the pipeline frees each before loading the next)
so two ~1.5B float32 models never sit in memory together.
"""

from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model registry for the minimal local case (all Apache-2.0, ungated, instruct).
#   same-family axis: M and J_same are the SAME Qwen model (self-judge) -> the
#   strongest place to look for a shared blind spot.
#   cross-family axis: J_cross is a different family at matched (~1.5-1.7B) size.
MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    # cross-family judge candidates (ungated, instruct, different family from Qwen)
    "granite-2b": "ibm-granite/granite-3.1-2b-instruct",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",  # gated; needs HF token
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LM:
    def __init__(self, key_or_id: str, device: str | None = None, dtype=torch.float32):
        self.model_id = MODELS.get(key_or_id, key_or_id)
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    # -- generation -------------------------------------------------------
    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy -> reproducible
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # -- perplexity (familiarity proxy, spec §4) --------------------------
    @torch.no_grad()
    def perplexity(self, context_messages: list[dict], response_text: str) -> float:
        """Teacher-forced perplexity of `response_text` given `context_messages`
        as the chat context. Lower = the model finds the text more familiar."""
        prompt = self.tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        resp_ids = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids
        if resp_ids.shape[1] == 0:
            return float("nan")
        input_ids = torch.cat([prompt_ids, resp_ids], dim=1).to(self.device)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100  # score only the response tokens
        out = self.model(input_ids=input_ids, labels=labels)
        # out.loss is mean NLL over scored tokens
        return float(math.exp(out.loss.item()))

    def close(self):
        del self.model
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
