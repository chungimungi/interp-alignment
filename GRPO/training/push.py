import os

import modal

from .app import app, image, model_out_vol
from .config import OUTPUT_DIR


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv(path=".env")],
    volumes={"/root/outputs": model_out_vol},
)
def push_from_volume(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str | None = None,
) -> str:
    from huggingface_hub import HfApi, login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    login(token=os.environ["HF_TOKEN"])

    pushed_urls = []

    # Push adapter (or full model if no merged variant requested)
    print(f"Loading model from {OUTPUT_DIR} ...")
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    HfApi().create_repo(repo_id=repo_id, private=private, exist_ok=True)
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    pushed_urls.append(f"https://huggingface.co/{repo_id}")
    print(f"Pushed adapter/model to {repo_id}")

    if push_merged:
        merged_dir = f"{OUTPUT_DIR}-merged"
        dense_repo_id = merged_repo_id or f"{repo_id}-merged"

        print(f"Loading merged model from {merged_dir} ...")
        merged_model = AutoModelForCausalLM.from_pretrained(merged_dir)
        merged_tokenizer = AutoTokenizer.from_pretrained(merged_dir)

        HfApi().create_repo(repo_id=dense_repo_id, private=private, exist_ok=True)
        merged_model.push_to_hub(dense_repo_id)
        merged_tokenizer.push_to_hub(dense_repo_id)
        pushed_urls.append(f"https://huggingface.co/{dense_repo_id}")
        print(f"Pushed merged model to {dense_repo_id}")

    return "Pushed: " + ", ".join(pushed_urls)


@app.local_entrypoint()
def push(
    repo_id: str,
    private: bool = False,
    push_merged: bool = True,
    merged_repo_id: str = "",
) -> None:
    resolved_merged_repo_id = merged_repo_id.strip() or None
    message = push_from_volume.remote(
        repo_id=repo_id,
        private=private,
        push_merged=push_merged,
        merged_repo_id=resolved_merged_repo_id,
    )
    print(message)
