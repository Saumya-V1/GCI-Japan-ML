# NFL Draft Prediction

## Overview

This project predicts whether an NFL prospect will be drafted based on college performance, physical measurements, NFL Combine results, and player background information.

The model uses an ensemble of CatBoost and LightGBM classifiers combined with extensive feature engineering and cross-validation to generate robust draft probability predictions.

## Features

### Data Processing

* Missing value handling
* Categorical feature encoding
* School prestige estimation using player frequency counts
* Combine participation indicators

### Feature Engineering

#### Athletic Metrics

* BMI
* Speed Score
* Burst Score
* Agility Score
* Power Ratio

#### Positional Features

For each position group:

* Percentile rankings
* Z-score normalization

#### Missingness Features

* Number of missing combine measurements
* Speed drill participation indicators

### Modeling

#### Base Models

* CatBoost Classifier
* LightGBM Classifier

#### Validation Strategy

* 10-Fold Stratified Cross Validation
* 5 Random Seeds per Fold
* Out-of-Fold (OOF) evaluation

#### Ensemble Method

Weighted blending:

* CatBoost: 85%
* LightGBM: 15%

The repository also includes a rank-blended version that combines percentile-ranked model outputs before blending.

## Dataset

The project expects:

```text
train.csv
test.csv
```

### Target Variable

| Column  | Description                              |
| ------- | ---------------------------------------- |
| Drafted | 1 if the player was drafted, 0 otherwise |

### Required Features

Examples include:

* Height
* Weight
* Sprint_40yd
* Vertical_Jump
* Broad_Jump
* Bench_Press_Reps
* Agility_3cone
* Shuttle
* Position
* Position_Type
* Player_Type
* School

## Installation

```bash
pip install pandas numpy scikit-learn lightgbm catboost scipy
```

## Usage

Run the training script:

```bash
python competition.py
```

The script will:

1. Load training and test data
2. Generate engineered features
3. Train CatBoost and LightGBM ensembles
4. Evaluate Out-of-Fold AUC
5. Generate test predictions
6. Save the submission file

## Output

Generated submission file:

```text
submission_0848_purified.csv
```

or

```text
submission_0848_rank_blended.csv
```

Format:

```csv
Id,Drafted
1,0.8234
2,0.1458
3,0.6789
```

## Model Architecture

```text
Feature Engineering
        ↓
10-Fold Stratified CV
        ↓
CatBoost Ensemble (5 Seeds)
        ↓
LightGBM Ensemble (5 Seeds)
        ↓
Weighted Blend (85/15)
        ↓
Draft Probability Prediction
```

## Performance

The ensemble achieved approximately:

* OOF ROC-AUC ≈ 0.848+

using engineered athletic and positional features combined with multi-seed model averaging.

## Future Improvements

* Optuna hyperparameter optimization
* Stacking with meta-models
* Additional college production metrics
* Historical draft trend features
* Advanced feature selection
* Deep learning tabular models

## Author

Developed as a machine learning solution for NFL Draft prediction using gradient boosting ensembles and advanced feature engineering.
