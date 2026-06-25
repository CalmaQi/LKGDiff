# LKGDiff

Latent knowledge graph generation framework for TSP with latent diffusion modeling.

## Structure

- `config/` – YAML configuration files for training stages
  - `stage1.yaml` – main stage-1 config
  - `stage1_wiki143.yaml`, `stage1_wiki79.yaml` – dataset-specific variants
- `script/` – training and evaluation scripts
  - `run.py` – standard training pipeline
  - `run_diff.py` – diffusion-based training pipeline
- `ultra/` – core model implementations
  - `stage12.py` – two-stage KG embedding framework
  - `diffnew1.py` – latent diffusion model components

## Usage

1. **Stage 1** – Train the base KG embedding model with the default config:
   ```bash
   python script/run.py --config config/stage1.yaml
   ```

2. **Stage 2** – Resume from the Stage 1 checkpoint for diffusion-based training:
   - Edit `config/stage1.yaml`
   - Set `checkpoint: <path_to_stage1_ckpt>`
   - Run:
   ```bash
   python script/run_diff.py --config config/stage1.yaml
   ```

> **Note:** Both stages share the same config file. Only the `checkpoint` field needs to be updated before Stage 2.

## Requirements

- Python 3.8+
- PyTorch
- See individual scripts for additional dependencies

## Citation

If you use this code, please cite the corresponding paper.
