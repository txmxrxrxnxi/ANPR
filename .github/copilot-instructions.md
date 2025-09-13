# Copilot Instructions for ANPR Codebase

## Overview
This project implements an Automatic Number Plate Recognition (ANPR) system. The codebase is organized for modularity and experimentation with machine learning models, data processing, and visualization.

## Architecture & Key Components
- **machines/**: Main source code. Contains:
  - `main.py`: Entry point for running the ANPR pipeline.
  - `model.py`, `train.py`, `evaluate.py`: Model definition, training, and evaluation logic.
  - `dataset.py`: Data loading and preprocessing.
  - `vizualize_model.py`: Model visualization utilities.
  - `gui.py`: (empty or placeholder) for GUI integration.
- **data/**: Contains training, validation, and test datasets in JSON and image formats.
- **older_models/**: Stores previous model checkpoints for reproducibility.
- **history/**: Training history and metrics (CSV, PNG).

## Developer Workflows
- **Training**: Run `machines/train.py` to train models. Model checkpoints are saved in the root or `older_models/`.
- **Evaluation**: Use `machines/evaluate.py` to assess model performance.
- **Visualization**: Use `machines/vizualize_model.py` to generate visualizations (see `history/`).
- **Data**: Data is loaded from `data/` using logic in `machines/dataset.py`.

## Patterns & Conventions
- **Model Checkpoints**: Saved as `.pth` files, named by version or date.
- **Metrics Logging**: Training metrics are logged to CSV and PNG in `history/`.
- **Data Format**: JSON files for splits, images for samples. Paths are relative to `data/`.
- **Modularity**: Each major function (train, evaluate, visualize) is a separate script for clarity and experimentation.

## Integration Points
- **PyTorch**: Used for model definition and training.
- **No explicit build system**: Scripts are run directly (e.g., `python machines/train.py`).
- **No test suite detected**: Add tests in `machines/` if needed.

## Example Workflow
```powershell
# Train a model
python machines/train.py

# Evaluate a model
python machines/evaluate.py

# Visualize training history
python machines/vizualize_model.py
```

## Recommendations for AI Agents
- Prefer editing scripts in `machines/` for core logic changes.
- When adding new models, save checkpoints in `older_models/` and update documentation.
- Keep data paths relative and consistent with `dataset.py` conventions.
- Log metrics to `history/` for reproducibility.
- If adding a GUI, implement in `machines/gui.py` and document usage.

## References
- See `README.md` for project summary.
- Key files: `machines/main.py`, `machines/model.py`, `machines/train.py`, `machines/evaluate.py`, `machines/dataset.py`, `machines/vizualize_model.py`.
