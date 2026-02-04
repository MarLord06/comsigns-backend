FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY comsigns-backend/requirements.txt /app/comsigns-backend/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/comsigns-backend/requirements.txt

# Copy the rest of the application
COPY comsigns-backend/ /app/comsigns-backend/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV COMSIGNS_DEVICE=cpu

# Expose port
EXPOSE 8080

# Start command
CMD ["sh", "-c", "cd /app/comsigns-backend && uvicorn backend.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
