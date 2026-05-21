# Gunakan image Python ringan
FROM python:3.10-slim

# Setup direktori kerja
WORKDIR /app

# Install dependency sistem untuk OpenCV (dibutuhkan oleh YOLO/Ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh sisa file (server.py, model .pt, .json)
COPY . .

# Berikan akses izin ke folder
RUN chmod -R 777 /app

# Jalankan server FastAPI di port 7860 (Port wajib dari Hugging Face)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
