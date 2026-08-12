FROM python:3.11-slim

WORKDIR /app

# system deps needed by matplotlib/shap wheels on slim images, plus curl for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build-time: regenerate all data/model artifacts inside the image so the
# container is fully self-contained and doesn't depend on files that only
# happen to exist on the host machine.
RUN python3 -m src.eda \
    && python3 -m src.split_data \
    && python3 -m src.feature_analysis \
    && python3 -m src.train \
    && python3 -m src.evaluate \
    && python3 -m src.stability_analysis \
    && python3 -m src.feature_tradeoff_analysis \
    && python3 -m src.shap_explain

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
