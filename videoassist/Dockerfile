FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
# Render injects $PORT; default 8000 for local `docker run`
CMD ["sh", "-c", "uvicorn videoassist.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
