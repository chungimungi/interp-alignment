"""
Sweep expansion factor / top-k for a fixed base vs aligned LLM setup.
"""
from . import config
from .activations import extract_activations_llm
from .train import train_crosscoder
from .utils import (
    flush_gpu,
    get_activations_dir,
    get_results_dir,
    load_activations,
    save_activations,
    save_json,
    set_seed,
)

# Example sweep grid; aligned checkpoint and base id should match your training run.
BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"
ALIGNED_MODEL = "/path/to/aligned_or_adapter"
ALIGNED_RUN_ID = "sweep"
LAYER = 15
POSITION = config.POSITION_LAST_PROMPT
DATASET_NAME = config.PREFERENCE_DATASET_NAME

SWEEP: tuple[int, int] = [(4, 200), (8, 400)]


def run_sweep():
    set_seed()
    results_root = config.CROSSCODER_RESULTS_DIR

    for expansion, topk in SWEEP:
        run_tag = f"{ALIGNED_RUN_ID}_ef{expansion}_k{topk}"
        results_dir = get_results_dir(BASE_MODEL, run_tag, LAYER, POSITION, base_dir=results_root)
        activations_dir = get_activations_dir(results_dir)
        act_path = activations_dir / "activations.pt"

        if not act_path.exists():
            print(f"Extracting activations -> {results_dir}")
            data = extract_activations_llm(
                base_model_id=BASE_MODEL,
                aligned_model_path=ALIGNED_MODEL,
                aligned_run_id=run_tag,
                layer=LAYER,
                position=POSITION,
                dataset_name=DATASET_NAME,
                max_prompt_tokens=config.MAX_PROMPT_TOKENS,
                trust_remote_code=False,
                hf_token=None,
            )
            save_activations(data, act_path)
            save_json(
                {
                    "base_model": BASE_MODEL,
                    "aligned_model": ALIGNED_MODEL,
                    "aligned_run_id": run_tag,
                    "layer": LAYER,
                    "position": POSITION,
                },
                results_dir / "run_meta.json",
            )
        else:
            data = load_activations(act_path)

        input_dim = int(data.get("hidden_size", data["activations_base"].shape[1]))
        print(f"Training ef={expansion} topk={topk} -> {results_dir}")
        train_crosscoder(
            activations_data=data,
            input_dim=input_dim,
            base_model_id=BASE_MODEL,
            aligned_run_id=run_tag,
            layer=LAYER,
            position=POSITION,
            results_dir=results_dir,
            expansion_factor=expansion,
            topk=topk,
        )
        flush_gpu()


if __name__ == "__main__":
    run_sweep()
