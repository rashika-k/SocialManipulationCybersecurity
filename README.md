# SocialManipulationCybersecurity

Python project for detecting social-engineering attacks and estimating manipulation risk from conversation text.

## What This Script Does

Running `main.py` performs four main tasks:

- Loads and cleans the dataset (`messages`, `is_attack`, `manipulation_type` required).
- Trains/evaluates attack detection models (Naive Bayes, Decision Tree, Random Forest) with:
  - multiple train/test splits (`70/30`, `80/20`, `90/10`)
  - 10-fold cross-validation
- Trains risk regression models from text (`RandomForestRegressor`, `DecisionTreeRegressor`, `XGBRegressor`) to estimate a `0.0-1.0` risk score.
- Starts an interactive terminal loop where you can enter custom messages and get:
  - risk score + risk level (`Low`, `Medium`, `High`)
  - predicted `manipulation_type` from multiple classifiers + majority vote

## Requirements

- Python 3.9+
- Packages in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

If XGBoost fails on macOS due to OpenMP, install:

```bash
brew install libomp
```

## Dataset Path

`main.py` currently reads:

`Data/social_cyber_engineering_dataset.csv`

Ensure that file exists and includes at least:

- `messages`
- `is_attack`
- `manipulation_type`

## Run

```bash
python3 main.py
```

## Interactive Usage

After training/evaluation logs print, the script prompts:

`Enter a message to analyze (or 'exit' to quit):`

Type any message to see risk + manipulation predictions, or type `exit` to stop.

## Notes

- Metrics are printed to terminal; the script does not currently save model artifacts.
- If you switch datasets (for example `output.csv`), make sure column names match what `main.py` expects.