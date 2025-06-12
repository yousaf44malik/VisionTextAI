# 🚀 VisionText AI

**VisionText AI** is a modular, containerized application built with FastAPI, combining the power of Computer Vision (CV) and Natural Language Processing (NLP) into a single, extensible system. It serves both as a learning sandbox and a scalable foundation for future multi-modal AI services.

---

## 🎯 Project Goals

1. Practice clean software engineering workflows using Git, Docker, and FastAPI.
2. Build a flexible, modular AI application that supports both language and vision tasks.
3. Serve as a starting point for developing multi-modal AI systems using open-source models.

---

## 🧠 Key Features

- 🧾 **/process-nlp** — NLP endpoint using EmBEL v1 or other MTEB-compatible models.
- 🖼️ **/process-image** — Vision endpoint using models like InternVL or BLIP-2.
- 🔄 **/query-text-to-image** — Hybrid endpoint that accepts natural language and queries vision models.
- 🐳 Docker-based local development and deployment.
- 📚 Built-in Git workflow for clean branching, tagging, and PR-based collaboration.
- 📊 Model benchmarking capabilities using the MTEB leaderboard.

---

## 📦 Tech Stack

| Category        | Tools/Frameworks                      |
|----------------|----------------------------------------|
| Language        | Python 3.11                           |
| API Framework   | FastAPI                               |
| NLP Models      | EmBEL v1, MTEB benchmark models        |
| Vision Models   | InternVL, BLIP-2, LLaVA                |
| Containerization| Docker (multi-stage builds)           |
| Testing Tools   | Swagger UI, Postman                   |
| Version Control | Git + GitHub                          |

---

## 🔧 API Endpoints

### 1. `POST /process-nlp`
```json
{
  "text": "Dental caries treatment options"
}
