# src/Dockerfile

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy everything (src/, main.py, configs/, requirements.txt)
COPY ../requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ../main.py ./main.py
COPY . ./src
COPY ../configs ./configs

EXPOSE 7860

CMD ["python", "main.py"]
