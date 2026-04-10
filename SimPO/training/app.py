import modal

app = modal.App("simpo-smollm3-training")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu24.04")
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
        "pynvml",
    )
    .add_local_python_source("training")
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
model_out_vol = modal.Volume.from_name("simpo-model-outputs", create_if_missing=True)
