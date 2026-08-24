"""Shared prompt assembly — optional reasoning-off (enable_thinking=False).

Our generation is completion-style (a prompt that ends with a code "prime" the model
continues). Reasoning models (Qwen3.x) emit <think>...</think> anyway, which pollutes
the output. enable_thinking=False is a *chat-template* feature, so to use it while
keeping our completion prime we wrap the instruction context in the chat template (with
an empty <think></think> block) and then append the code prime as the start of the
assistant turn:

    <|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prime}

Only applied when the tokenizer's chat template actually supports enable_thinking
(Qwen3); otherwise we fall back to plain completion `{context}\n{prime}`.
"""


def supports_no_think(tokenizer) -> bool:
    ct = getattr(tokenizer, "chat_template", None)
    return bool(ct) and "enable_thinking" in ct


# Reasoning models whose thinking cannot be switched off through `enable_thinking` are handled
# by prefilling an already-closed think block, so generation starts on the answer with zero
# thinking tokens. Without it they spend the whole budget thinking and get truncated.
_THINK_PREFILL = ("I have done a short CoT reasoning and Aha! I have found the right answer."
                  "\n</think>\n\n")


def apply_think_prefill(prompt: str) -> str:
    """If the template already opens <think>, append the closing part; else prefill both."""
    if prompt.rstrip().endswith("<think>"):
        return prompt + "\n" + _THINK_PREFILL
    return prompt + "<think>\n" + _THINK_PREFILL


def build_chat_prompt(tokenizer, system: str, user: str, no_think: bool = True) -> str:
    """With THINK_PREFILL=1 in the environment, apply the think prefill automatically,
    so a reasoning model runs through the same chain without touching any call site."""
    """Chat prompt via apply_chat_template, with robust fallbacks:
      - tokenizer has no chat_template      -> plain text "{system}\\n\\n{user}"
      - enable_thinking=False applied        -> only when the template supports it
      - system role unsupported (e.g. gemma) -> retry as a single combined user message
    """
    if not getattr(tokenizer, "chat_template", None):
        return f"{system}\n\n{user}\n"
    import os
    _pf = os.environ.get("THINK_PREFILL") == "1"
    kw = {"enable_thinking": False} if (no_think and supports_no_think(tokenizer)) else {}
    for msgs in (
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        [{"role": "user", "content": f"{system}\n\n{user}"}],
    ):
        try:
            out = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **kw)
            return apply_think_prefill(out) if _pf else out
        except Exception:
            continue
    return f"{system}\n\n{user}\n"


def assemble_prompt(tokenizer, context: str, prime: str, no_think: bool) -> str:
    """context = instruction + problem (+ contract info); prime = the code prime the
    model should continue (e.g. 'def f(...):\\n    ')."""
    if no_think and supports_no_think(tokenizer):
        head = tokenizer.apply_chat_template(
            [{"role": "user", "content": context}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return head + prime
    return f"{context}\n{prime}"
