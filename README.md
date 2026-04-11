# Student Performance Prediction with MLOps Pipeline

This project predicts whether a student will pass or fail using a simple machine learning pipeline built with DVC, MLflow, FastAPI, Docker, and GitHub Actions.

## Project Goal

- Version the dataset with DVC
- Train a logistic regression model
- Log experiments in MLflow
- Serve predictions through FastAPI
- Containerize the API with Docker
- Re-train the model when the dataset changes
- Automate the workflow with CI/CD

## Folder Structure

```text
student-mlops/
├── data/
│   └── student.csv
├── training/
│   └── train.py
├── app/
│   └── main.py
├── dvc.yaml
├── dvc.lock
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset

The dataset is intentionally small and simple:

- `study_hours`
- `attendance`
- `previous_marks`
- `pass` target label

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Initialize Git and DVC if you are starting from scratch:

```bash
git init
dvc init
```

Track the dataset with DVC:

```bash
dvc add data/student.csv
```

## Training

Run the training script directly:

```bash
python training/train.py
```

Or run the DVC pipeline:

```bash
dvc repro
```

The trained model is saved to `app/model.pkl` and the run is logged in MLflow.

## MLflow

Start the MLflow UI:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

## API

Start the FastAPI application:

```bash
uvicorn app.main:app --reload
```

Open the docs:

```text
http://localhost:8000/docs
```

Example request:

```json
{
  "study_hours": 4,
  "attendance": 75,
  "previous_marks": 65
}
```

Example response:

```json
{
  "prediction": "Pass",
  "probability": 0.87
}
```

## Docker

Build and run the container:

```bash
docker build -t student-api .
docker run -p 8000:8000 student-api
```

## CI/CD

The GitHub Actions workflow in `.github/workflows/ci.yml` installs dependencies, initializes DVC if needed, runs `dvc repro`, and builds the Docker image.

## Auto Retraining

When `data/student.csv` changes, rerun:

```bash
dvc repro
```

This retrains the model and updates `app/model.pkl`.

## Viva Summary

This project implements an end-to-end MLOps pipeline where data is versioned using DVC, models are trained and tracked using MLflow, deployed via FastAPI in Docker containers, and automated using CI/CD pipelines. The system supports continuous retraining when new data is introduced.
