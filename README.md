# interp-alignment

## Batch Top-K SAE training

This repository now includes a convenience script for training a Batch Top-K sparse autoencoder using [sae-lens](https://github.com/jbloomAus/SAELens).

1. Install training dependencies (sae-lens already includes transformer-lens):
   ```bash
   pip install "sae-lens[train]"
   ```
2. Run the trainer with your preferred model, dataset, and hook:
   ```bash
   python batch_topk_sae.py \
     --model-name gpt2-small \
     --dataset roneneldan/TinyStories \
     --hook-name blocks.0.hook_mlp_out \
     --k 64 \
     --d-sae 4096 \
     --training-tokens 200000 \
     --device auto
   ```

Key flags:
- `--model-name` / `--model-class-name`: target model (HookedTransformer by default).
- `--dataset`: Hugging Face dataset to stream; set `--disable-streaming` to materialize locally.
- `--hook-name`: activation hook to train on (e.g., `blocks.6.hook_resid_pre`).
- `--k`: average number of active SAE features across the batch.
- `--output-path`: where the inference SAE checkpoint is written.
- Add `--log-to-wandb` (with optional `--wandb-project` / `--wandb-entity`) to enable tracking.
