# End-to-End MLOps Pipeline (Iris Classification)

## 📖 Project Overview

This project demonstrates how a **machine learning model is built, packaged, and served as a running system**, rather than remaining as a notebook or standalone script.

Using the **Iris classification problem** as a simple and well‑understood example, the project focuses on the *engineering side* of ML systems:

* Training a reproducible ML model
* Tracking experiments and artifacts using MLflow
* Serving predictions through a REST API
* Packaging the application with Docker
* Deploying the service to Kubernetes (AKS)
* Exposing basic system and application metrics

The ML problem itself is intentionally simple so that the attention stays on **system design, deployment, and operability**.

---

## 🎯 Core Intention

The core intention of this project is to show that:

> **A trained ML model only becomes useful when it can run reliably as a service.**

Instead of optimizing algorithms, this project emphasizes:

| Focus Area | What This Project Demonstrates          |
| ---------- | --------------------------------------- |
| Training   | Repeatable, script‑driven training      |
| Tracking   | Versioned experiments and models        |
| Serving    | API‑based model access                  |
| Deployment | Containerized, Kubernetes‑based runtime |
| Operations | Health checks and basic observability   |

The project treats the model as **software that must be built, deployed, and operated**, not as a one‑time experiment.

---

## 🌸 Why the Iris Dataset?

The Iris dataset is used deliberately because it is:

* Small and fast to train
* Easy to understand without ML expertise
* Free from heavy data‑engineering requirements

This keeps the focus on **how the system works**, not on dataset complexity. The same architecture can be reused for larger datasets without changing the overall design.

---

## 🧠 End‑to‑End Flow

```
Load Data
   ↓
Train Model
   ↓
Log Metrics & Artifacts (MLflow)
   ↓
Serve Model via FastAPI
   ↓
Package with Docker
   ↓
Deploy to Kubernetes (AKS)
   ↓
Expose Metrics & Health Endpoints
```

At runtime, external users or services interact only with the **API**, not directly with the model or training code.

---

## 🏗️ High‑Level Architecture

```
Training Script
   │
   ▼
MLflow (Experiments & Artifacts)
   │
   ▼
FastAPI Application
   │
   ▼
Docker Image
   │
   ▼
Kubernetes (AKS)
   │
   ▼
Prometheus → Grafana
```

---

## 📁 Project Structure Explained

```
MLOps-End-To-End-Pipeline/
│
├── api/                     # FastAPI application (model serving)
│   └── main.py              # API endpoints, health, metrics
│
├── src/                     # Core ML logic
│   ├── data_loader.py       # Dataset loading
│   ├── preprocessing.py     # Feature preprocessing
│   ├── training.py          # Model training logic
│   ├── model_registry.py    # MLflow model registration
│   └── drift_detector.py    # Drift detection logic
│
├── scripts/                 # Pipeline automation
│   ├── train_pipeline.py    # End-to-end training pipeline
│   └── model_promotion.py   # Model promotion logic
│
├── tests/                   # Unit & API tests
│   ├── test_api.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_training.py
│
├── k8s/                     # Kubernetes manifests
│   ├── app/                 # API deployment & service
│   └── observability/       # Prometheus, Grafana, MLflow
│
├── .github/workflows/       # GitHub Actions CI/CD pipelines
│   ├── tests.yml            # Run tests
│   ├── build.yml            # Build & push Docker image
│   ├── deploy.yml           # Deploy to AKS
│   └── observability.yml    # Deploy monitoring stack
│
├── Dockerfile               # Container definition
├── docker-compose.yml       # Local multi-service setup
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Development & testing dependencies
└── README.md
```

---

## 🔬 Machine Learning Details

| Aspect       | Description                               |
| ------------ | ----------------------------------------- |
| Problem Type | Multiclass classification                 |
| Dataset      | Iris dataset                              |
| Library      | scikit‑learn                              |
| Metrics      | Accuracy and basic classification metrics |

The ML code is intentionally **simple and modular**, making it easy to replace the model without changing the system architecture.

---

## 🔁 Training & Experiment Tracking

Training is executed using a Python script:

```bash
python scripts/train_pipeline.py
```

During training:

* Parameters and metrics are logged to **MLflow**
* The trained model artifact is stored for later use

This enables reproducibility and comparison between runs.

---

## 🌐 Model Serving (FastAPI)

The FastAPI service exposes:

* `POST /predict` – returns model predictions
* `GET /health` – basic service health information
* `GET /metrics` – Prometheus‑compatible metrics

The API loads the trained model at startup and serves predictions over HTTP.

---

## 🐳 Containerization

The application is packaged using Docker to ensure consistent runtime behavior:

```bash
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

---

## ☸️ Kubernetes Deployment (AKS)

The containerized API is deployed to **Azure Kubernetes Service (AKS)** using Kubernetes manifests.

Kubernetes handles:

* Running the API pods
* Restarting failed containers
* Exposing the service via a Kubernetes Service

---

## 📊 Monitoring & Observability

* **Prometheus** scrapes metrics from the `/metrics` endpoint
* **Grafana** visualizes request counts, latency, and service health

This provides basic visibility into how the system behaves after deployment.

---

## 🔄 CI/CD with GitHub Actions

GitHub Actions automate:

* Running tests on every change
* Building Docker images
* Deploying updated versions to AKS

This ensures that changes are validated and deployed in a repeatable way.

---

## ▶️ Running Locally

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
uvicorn api.main:app --reload
```

---

## 🔮 Future Improvements

* Automated retraining pipelines
* More advanced drift detection
* Canary or blue‑green deployments
* Feature store integration
* Model explainability dashboards

---

## 👤 Author

This project was built to demonstrate **practical MLOps system design**, focusing on clarity, correctness, and real‑world deployment patterns.

---

⭐ **Key takeaway:** this repository shows how a simple ML model can be transformed into a reliable, observable production service.
