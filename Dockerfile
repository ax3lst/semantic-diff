FROM python:3.11-slim

# system & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download de_core_news_sm

COPY . .

# Gunicorn als Prod-Server
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
