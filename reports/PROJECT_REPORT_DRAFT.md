# Project Report Draft - Social Engineering Manipulation Type Classification

## A) Attribute Selection and Justification

### Target Variable
- `manipulation_type` (14 classes).

### Included Attributes
- Text: `messages` -> TF-IDF (`max_features=2500`, unigrams).
- Categorical: `is_attack`, `communication_channel`, `relationship_context`, `organizational_pretext`, `employee_attachment_style`, `escalation_pattern`.
- Numeric: `conversation_length`, `attack_intensity`, `victim_confusion_score`, `pressure_score`, `urgency_score`, `threat_level`, `self_doubt_score`, `employee_extraversion`, `employee_emotional_resilience`, `employee_assertiveness`, `avg_response_delay_seconds`, `word_count_total`, `question_count`, `denial_count`.

### Discarded Attributes and Why
- `conversation_id`, `original_conversation_id`: identifiers only.
- `source_dataset`: possible source leakage and not behavioral content.

## B) Missing Attributes and Fix
- Missing values found:
  - `communication_channel` (10,000)
  - `relationship_context` (10,000)
  - `organizational_pretext` (10,000)
- Handling (without editing raw CSV):
  - Categorical imputation: constant `missing`
  - Numeric imputation: median

## C) Method and Rationale
- Decision Tree, Naive Bayes, Random Forest, Linear SVM.
- Rationale: balanced comparison across interpretable baseline, text baseline, ensemble model, and strong linear text classifier.

## D) Train/Test Sizes and 10-Fold CV
- Splits used: 90/10, 80/20, 70/30, 60/40 (stratified).
- CV: Stratified 10-fold with shuffle, random_state=42.

## 6) Classification Metrics

### Average Test Metrics Across Split Sizes
```text
       model  accuracy  precision_weighted  recall_weighted  f1_weighted  train_time_s  predict_time_s
   LinearSVM    1.0000              1.0000           1.0000       1.0000        5.8891          0.2856
RandomForest    1.0000              1.0000           1.0000       1.0000        6.8675          0.4180
DecisionTree    0.9956              0.9956           0.9956       0.9955        1.9358          0.6409
  NaiveBayes    0.9625              0.9690           0.9625       0.9639        1.4322          0.7071
```

### 10-Fold CV Metrics
```text
       model  accuracy_mean  precision_weighted_mean  recall_weighted_mean  f1_weighted_mean  accuracy_std  fit_time_mean_s  score_time_mean_s
RandomForest         1.0000                   1.0000                1.0000            1.0000        0.0000           5.9174             0.2200
   LinearSVM         1.0000                   1.0000                1.0000            1.0000        0.0000           9.0707             0.2020
DecisionTree         0.9964                   0.9965                0.9964            0.9964        0.0015           2.5071             0.2390
  NaiveBayes         0.9720                   0.9751                0.9720            0.9728        0.0031           0.8222             0.1167
```

## Interpretation
- Best split-run weighted F1: **RandomForest** (F1=1.0000, split=0.90/0.10).
- Best 10-fold CV weighted F1: **RandomForest** (mean F1=1.0000).

## Reproducibility
- Random seed locked at 42.
- Script: `analysis/run_experiments.py`.
- Outputs: `reports/model_results_splits.csv`, `reports/model_results_cv10.csv`, `reports/experiment_metadata.json`.
