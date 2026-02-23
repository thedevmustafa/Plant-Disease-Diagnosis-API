---
title: Plant Pathology API
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
startup_duration_timeout: 1h
---

# 🌿 Plant Disease Diagnosis Microservice

An AI-powered pathology detection engine designed to identify plant diseases from leaf imagery. This microservice uses a Deep Learning model (MobileNetV2) and provides actionable treatment protocols via an integrated SQLite database.

## 🚀 Features
* **AI Engine:** MobileNetV2 architecture trained on the PlantVillage dataset (38 classes).
* **Secure API:** Headless FastAPI backend protected via `X-API-KEY` header authentication.
* **Integrated Database:** Automated retrieval of organic and chemical control measures.
* **Live UI:** A built-in diagnostic dashboard for real-time testing.
* **Containerized:** Fully Dockerized for seamless deployment on cloud platforms like Koyeb.

---

## 🏗️ Architecture
The system is built as a **Headless Microservice**:
1.  **Frontend:** Static HTML/JS dashboard served via FastAPI.
2.  **Backend:** RESTful API built with Python and FastAPI.
3.  **Inference:** PyTorch implementation of MobileNetV2.
4.  **Storage:** SQLite database containing symptoms and control measures.
