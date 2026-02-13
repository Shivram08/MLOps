# Student Exam Score Prediction with DVC Pipeline

## Overview

This lab demonstrates how to use Data Version Control (DVC) to manage datasets, machine learning pipelines, and experiment tracking in a reproducible way.

Instead of the original placeholder dataset, this implementation uses the Student Performance Dataset (5000 records) and builds a full regression pipeline to predict:

Target Variable: exam_score

The project integrates:
- DVC for data and artifact versioning
- Google Cloud Storage (GCS) as remote storage
- A multi-stage ML pipeline (prepare → train → evaluate)
- Experiment tracking via Git + DVC

---

## Dataset

Source:
https://www.kaggle.com/datasets/amar5693/student-performance-dataset

The dataset includes:
- Demographics (age, gender, academic_level)
- Study habits (study_hours, self_study_hours, etc.)
- Digital behavior (screen time, gaming, social media)
- Health & lifestyle indicators
- Mental health metrics
- Academic outputs (focus_index, burnout_level, productivity_score, exam_score)

For this lab, we focus on predicting:
exam_score

---

## Project Structure
```
Lab_1/
│
├── data/
│   ├── student_performance.csv.dvc
│   └── processed/
│
├── models/
│
├── src/
│   ├── prepare.py
│   ├── train.py
│   └── evaluate.py
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── metrics.json
└── README.md
```
---

## Pipeline Stages

### 1. Prepare

- Cleans column names
- One-hot encodes categorical variables
- Splits dataset into train/test sets

Outputs:
- data/processed/X_train.csv
- data/processed/X_test.csv
- data/processed/y_train.csv
- data/processed/y_test.csv

Command:
python src/prepare.py

---

### 2. Train

- Trains a RandomForestRegressor
- Hyperparameters are defined in params.yaml

Outputs:
- models/model.pkl

Command:
python src/train.py

---

### 3. Evaluate

- Loads trained model
- Evaluates on test set
- Computes RMSE and R²

Outputs:
- metrics.json

Command:
```bash
python src/evaluate.py
```
---

## DVC Pipeline

To reproduce the entire workflow:

```bash
dvc repro
```
To push artifacts to remote storage:
```bash
dvc push
```
To pull artifacts:
```bash
dvc pull
```
---

## Experiment Tracking

Metrics are versioned with DVC and committed via Git.

Example experiment comparison:
```bash
dvc metrics diff HEAD~1 HEAD --targets Labs/Data_Labs/DVC_Labs/Lab_1/metrics.json
```
Example output:
![Metrics Diff](assets\metrics.png)
This demonstrates how hyperparameter changes impact model performance.

---

## Remote Storage (GCS)

Artifacts and datasets are stored in a Google Cloud Storage bucket via DVC.

Service account credentials are NOT included in this repository for security reasons.

To configure remote access locally:
```bash
dvc remote add -d studentgcs gs://<your-bucket-name>
dvc remote modify studentgcs credentialpath <path-to-your-json-key>
```
![Bucket](assets\Bucket.png)

---

## Reproducibility

To reproduce this project from scratch:
```bash
git clone <repo-url>
cd Labs/Data_Labs/DVC_Labs/Lab_1
pip install -r requirements.txt
dvc pull
dvc repro
```
This will:
- Download versioned data and model artifacts
- Recreate the exact pipeline outputs
- Regenerate metrics.json


---

## Conclusion

This lab demonstrates a fully reproducible ML workflow using DVC with:

- Version-controlled dataset
- Cloud-backed artifact storage
- Structured pipeline
- Experiment comparison capabilities

The entire workflow can be reproduced using:
```bash
dvc repro
```