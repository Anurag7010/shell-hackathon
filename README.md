# Shell.ai Hackathon - Fuel Blend Property Prediction

This repository contains my final hackathon workspace for multi-target regression on fuel blend properties.

The project predicts 10 target values (`BlendProperty1` to `BlendProperty10`) from component fractions and component properties.

## Repository Layout

- `shell_10.py`: Complete end-to-end pipeline with feature engineering, training, and submission generation.
- `shell_1_1.py`: Target-wise optimization approach.
- `shell_1_2.py`: Mathematical optimization approach.
- `shell_1_3.py`: Integrated optimization pipeline combining multiple strategies.
- `shell.ipynb`, `shell3.ipynb`, `shell4.ipynb`, `shell5.ipynb`, `shell_9.ipynb`: Experiment notebooks.
- `train.csv`, `test.csv`, `sample_solution.csv`: Hackathon dataset files.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run one of the pipelines (example):

```bash
python shell_10.py
```

## Notes

- Most scripts expect `train.csv` and `test.csv` in the project root.
- Output submissions are generated as CSV files in the project root when scripts are executed.
- `catboost` is optional in some scripts, but recommended for full functionality.

## Environment

- Python 3.10+
- macOS/Linux/Windows compatible (standard Python stack)
