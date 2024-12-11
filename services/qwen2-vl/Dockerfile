FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# COPY entfernt, da wir das Volume mounten

EXPOSE 8000
CMD ["python3", "main.py"]