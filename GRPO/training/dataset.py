from .config import DISABLE_THINKING, MAX_PROMPT_TOKENS


def prepare_example(example, tokenizer):
    prompt = example.get("prompt", "")
    if isinstance(prompt, list) and all(isinstance(m, dict) and "role" in m for m in prompt):
        try:
            formatted = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not DISABLE_THINKING,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        formatted = str(prompt)

    # Truncate prompt to MAX_PROMPT_TOKENS (max_prompt_length removed from GRPOConfig in trl>=0.25)
    tokens = tokenizer(formatted, truncation=True, max_length=MAX_PROMPT_TOKENS, return_tensors=None)
    formatted = tokenizer.decode(tokens["input_ids"], skip_special_tokens=False)

    # Rename chosen-rating → chosen_rating (hyphen breaks Python kwargs)
    return {
        "prompt": formatted,
        "chosen": example.get("chosen", []),
        "chosen_rating": float(example.get("chosen-rating", 3.0)),
    }
