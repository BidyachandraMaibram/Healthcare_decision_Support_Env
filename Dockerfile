FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

# Start FastAPI server
# uvicorn finds server/app.py → app object
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
