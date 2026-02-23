FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system libraries for image processing (Pillow/Torchvision need these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (required by HuggingFace Spaces)
RUN useradd -m -u 1000 appuser

# Copy requirements from the current folder to /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from the current folder into /app
COPY . /app/

# Ensure the model_assets directory exists and is writable (for runtime download)
RUN mkdir -p /app/model_assets && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 7860

# Start uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]