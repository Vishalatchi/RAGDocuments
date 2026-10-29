FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only PyTorch first (keeps the image small), then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/

# Run as non-root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# ASSUMPTION: FastAPI app object named `app` in api/main.py. Change if yours differs.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
