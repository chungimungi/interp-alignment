from .config import DISABLE_THINKING


def _to_text(value, tokenizer) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        is_chat_messages = all(isinstance(item, dict) and "role" in item for item in value)
        if is_chat_messages:
            try:
                return tokenizer.apply_chat_template(
                    value,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=not DISABLE_THINKING,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    value,
                    tokenize=False,
                    add_generation_prompt=False,
                )

        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        content = value.get("content", "")
        return str(content)
    return str(value)


def _as_messages(value):
    if isinstance(value, list) and all(
        isinstance(item, dict) and "role" in item and "content" in item for item in value
    ):
        return value
    return None


def normalize_example(example, tokenizer):
    """
    Normalize a preference example to {prompt, chosen, rejected} text format.
    Same normalization as the DPO script — CPOTrainer expects this format.
    """
    prompt_messages = _as_messages(example.get("prompt"))
    chosen_messages = _as_messages(example.get("chosen"))
    rejected_messages = _as_messages(example.get("rejected"))

    if prompt_messages and chosen_messages and rejected_messages:
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not DISABLE_THINKING,
            )
            chosen_full = tokenizer.apply_chat_template(
                prompt_messages + chosen_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=not DISABLE_THINKING,
            )
            rejected_full = tokenizer.apply_chat_template(
                prompt_messages + rejected_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=not DISABLE_THINKING,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen_full = tokenizer.apply_chat_template(
                prompt_messages + chosen_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            rejected_full = tokenizer.apply_chat_template(
                prompt_messages + rejected_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        if chosen_full.startswith(prompt_text) and rejected_full.startswith(prompt_text):
            return {
                "prompt": prompt_text,
                "chosen": chosen_full[len(prompt_text) :],
                "rejected": rejected_full[len(prompt_text) :],
            }

    prompt = _to_text(example.get("prompt", ""), tokenizer)
    chosen = _to_text(example["chosen"], tokenizer)
    rejected = _to_text(example["rejected"], tokenizer)

    if prompt and chosen.startswith(prompt) and rejected.startswith(prompt):
        return {
            "prompt": prompt,
            "chosen": chosen[len(prompt) :],
            "rejected": rejected[len(prompt) :],
        }

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
