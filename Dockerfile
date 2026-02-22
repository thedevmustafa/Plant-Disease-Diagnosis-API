FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system libraries for image processing (Pillow/Torchvision need these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the current folder to /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from the current folder into /app
COPY . /app/

EXPOSE 8000

# Start uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]