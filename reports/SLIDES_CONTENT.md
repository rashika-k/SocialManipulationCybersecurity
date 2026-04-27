# Slide Content - Social Engineering Attack Detection Project

## Slide 1 - Title
- Social Engineering Manipulation Type Detection
- Cyber Innovation Lab (Fall 2025)
- Team: Ryan C Mitchell, Rashika Karmacharya

## Slide 2 - Problem and Objective
- Classify manipulation techniques in communications.
- Build behavioral detection models for social engineering tactics.

## Slide 3 - Dataset
- Updated dataset: 20,000 rows, 25 columns.
- Target: `manipulation_type` (14 classes).

## Slide 4 - Features Used
- Text from `messages` via TF-IDF.
- Behavioral numeric indicators + communication context categories.
- Removed IDs/source fields to reduce noise/leakage.

## Slide 5 - Missing Data Strategy
- Missing in three categorical fields.
- In-pipeline imputation (categorical=`missing`, numeric=median).
- Raw dataset unchanged.

## Slide 6 - Models Compared
- Decision Tree
- Naive Bayes
- Random Forest
- Linear SVM

## Slide 7 - Experimental Setup
- Splits: 90/10, 80/20, 70/30, 60/40.
- Stratified 10-fold CV.
- Metrics: accuracy, precision, recall, weighted F1, runtime.
- Seed = 42 for reproducibility.

## Slide 8 - Key Results
- Best split F1 model: RandomForest (1.0000).
- Best CV F1 model: RandomForest (1.0000).

## Slide 9 - Conclusion
- Multiple models were evaluated under required split and CV settings.
- Strong candidate model identified using weighted F1.

## Slide 10 - Next Steps
- Per-class confusion analysis.
- Real-time scoring prototype.
- Training recommendation logic by vulnerability profile.
