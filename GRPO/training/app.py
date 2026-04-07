import modal

app = modal.App("grpo-smollm3-training")

image = (
    modal.Image.from_registry("python:3.11")
    .pip_install(
        "torch",
        "transformers>=4.45.0",
        "datasets",
        "trl>=0.11.0,<1.0.0",
        "peft",
        "accelerate",
        "bitsandbytes",
        "wandb",
        "huggingface_hub",
        "rouge_score",
        "pynvml",
    )
    .add_local_python_source("training")
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
model_out_vol = modal.Volume.from_name("grpo-model-outputs", create_if_missing=True)
