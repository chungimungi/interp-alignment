# interp-alignment

This repository contains tools and scripts for the interpretability of alignment in Large Language Models.

## Overview

The codebase is split into several main components:

- `alignment-scripts/`: Contains scripts for training models using alignment techniques like DPO, GRPO, KTO, PPO, and SimPO.
- `interp_utils/`: A collection of utility scripts for interpretability methods including:
  - `crosscoder-singlelayer`/`crosscoder-multilayer`: Analyzing crosscoders across layers.
  - `linear-probe/`: Layerwise contrastive logistic probes on preference datasets.
  - `sae/`: Scripts for evaluating and visualizing Sparse Autoencoders (SAEs).
- `sae-feature.py`: A main visualization tool used to analyze specific SAE features across different models.
- `sae_anchor_transfer.py`: Analysis of anchor features transferring across alignments.
- `plot_monosemantic_features.py`: Tools focusing on plotting monosemantic behaviors of specific features.

## Setup

First, be sure to install the requirements:
```bash
pip install -r requirements.txt
```

Citation:
```
@misc{sinha2026mechanisticanalysisalignmentalgorithms,
      title={Mechanistic Analysis of Alignment Algorithms in Language Models}, 
      author={Aarush Sinha and Ishan Garg and Veeraraju Elluru and Arth Singh and Kushal Garg},
      year={2026},
      eprint={2606.09850},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.09850}, 
}
```
