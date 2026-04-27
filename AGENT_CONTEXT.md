# Agent Context Log

This file is a living context log to avoid context rot across chat turns.
I will update it as requirements, decisions, and progress evolve.

## Project
- Course: Cyber Innovation Lab (Fall 2025), Difficulty Level 4
- Project: Social Engineering Attack Detection in Corporate Communications
- Repository: `SocialManipulationCybersecurity`

## Current Dataset Snapshot
- File: `data/cyber_social_engineering_dataset.csv`
- Rows: 10,000
- Columns: 22
- Header:
  - `conversation_id`
  - `manipulation_type`
  - `is_attack`
  - `communication_channel`
  - `organizational_pretext`
  - `conversation_length`
  - `attack_intensity`
  - `messages`
  - `victim_confusion_score`
  - `pressure_score`
  - `urgency_score`
  - `threat_level`
  - `self_doubt_score`
  - `employee_extraversion`
  - `employee_emotional_resilience`
  - `employee_assertiveness`
  - `employee_attachment_style`
  - `avg_response_delay_seconds`
  - `escalation_pattern`
  - `word_count_total`
  - `question_count`
  - `denial_count`

## Assignment Requirements Tracked
1. Attribute selection and justified discards.
2. Missing value detection and handling.
3. Method choice and rationale.
4. Multiple train/test splits + ten-fold cross-validation.
5. Train algorithms.
6. Evaluate:
   - Classification: accuracy, recall, precision, F1 (and efficiency as needed).
   - Regression (if used): MSE, RMSE, MAE, R^2.

## Open Questions
- "Depth type" selection for final submission packaging still open.

## Next Planned Deliverables
- Optional: confusion matrix visuals and per-class report if time allows.

## Confirmed Decisions (Apr 26, 2026)
- Target: `manipulation_type`.
- Stack: Python + scikit-learn.
- Feature approach: TF-IDF for text (`messages`) plus structured features.
- Dataset edits: Do not alter raw dataset without user approval.
- Reproducibility: lock random seed.
- Required evaluation: multiple train/test split sizes + ten-fold CV.
- Submission depth selected: Depth 2 (strong submission).

## Implementation Completed
- Added experiment script: `analysis/run_experiments.py`.
- Compared models:
  - Decision Tree
  - Naive Bayes (MultinomialNB)
  - Random Forest
  - Linear SVM
- Split sizes executed (stratified):
  - 90/10
  - 80/20
  - 70/30
  - 60/40
- Ten-fold CV executed:
  - StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
- Missing-value handling:
  - Categorical -> constant "missing" (in pipeline)
  - Numeric -> median (in pipeline)
  - Raw CSV unchanged

## Current Results Snapshot
- Dataset profiled: `data/social_cyber_engineering_dataset.csv` (20,000 rows, 25 cols)
- Missing values total: 30,000 (three categorical columns, 10,000 each)
- Best split-run weighted F1: RandomForest (1.0000)
- Best 10-fold CV weighted F1: RandomForest (1.0000)

## Generated Artifacts
- `reports/model_results_splits.csv`
- `reports/model_results_cv10.csv`
- `reports/experiment_metadata.json`
- `reports/PROJECT_REPORT_DRAFT.md`
- `reports/SLIDES_CONTENT.md`
