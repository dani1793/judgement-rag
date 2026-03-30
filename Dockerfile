FROM python:3.11-slim

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
